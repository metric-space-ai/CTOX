use crate::agentic::CompletionReviewDirective;
use crate::agentic::ExecSessionDirective;
use crate::agentic::choose_next_task_focus;
use crate::agentic::probe_kleinhirn_health;
use crate::agentic::run_agentic_task_once;
use crate::agentic::should_run_agentic_loop;
use crate::brain_runtime::apply_recommended_browser_vision_kleinhirn_upgrade;
use crate::brain_runtime::apply_targeted_kleinhirn_upgrade;
use crate::brain_runtime::attempt_kleinhirn_runtime_repair;
use crate::brain_runtime::extract_requested_local_kleinhirn_model;
use crate::brain_runtime::grosshirn_runtime_configured;
use crate::brain_runtime::local_kleinhirn_upgrade_available;
use crate::brain_runtime::prepare_grosshirn_activation_from_message;
use crate::browser_engine::inspect_browser_engine;
use crate::browser_engine::run_browser_action;
use crate::browser_subworkers::advance_browser_subworkers;
use crate::command_exec::read_session;
use crate::command_exec::snapshot_session;
use crate::command_exec::start_session;
use crate::command_exec::terminate_session;
use crate::command_exec::write_session;
use crate::context_controller::ContextPreparedArtifact;
use crate::context_controller::ContextPackagePreflightDecision;
use crate::context_controller::decide_context_package_preflight;
use crate::context_controller::prepare_context_package;
use crate::context_controller::prepare_context_package_with_trigger;
use crate::contracts::AgentState;
use crate::contracts::BrainModel;
use crate::contracts::GpuDevice;
use crate::contracts::LoopSafetyPolicy;
use crate::contracts::ModelTuneCandidate;
use crate::contracts::Paths;
use crate::contracts::SystemCensus;
use crate::contracts::append_boot_entry;
use crate::contracts::load_bios;
use crate::contracts::load_census;
use crate::contracts::load_context_optimization_policy;
use crate::contracts::load_homepage_policy;
use crate::contracts::load_loop_safety_policy;
use crate::contracts::load_model_policy;
use crate::contracts::load_organigram;
use crate::contracts::load_root_auth;
use crate::contracts::load_self_preservation_state;
use crate::contracts::now_iso;
use crate::contracts::path_display_name;
use crate::contracts::write_agent_state;
use crate::contracts::write_census;
use crate::desktop_session::detect_desktop_session_env;
use crate::runtime_db::apply_proactive_contact_validation;
use crate::runtime_db::arm_task_grosshirn_boost;
use crate::runtime_db::attach_dispatch_task_to_candidate;
use crate::runtime_db::attach_validation_task_to_candidate;
use crate::runtime_db::block_task;
use crate::runtime_db::complete_agent_turn;
use crate::runtime_db::complete_review_task;
use crate::runtime_db::complete_task;
use crate::runtime_db::continue_active_task;
use crate::runtime_db::delegate_task_to_worker;
use crate::runtime_db::emit_self_review_task;
use crate::runtime_db::expire_stale_grosshirn_boost;
use crate::runtime_db::grosshirn_boost_available;
use crate::runtime_db::has_open_loop_incident;
use crate::runtime_db::ingest_pending_loop_interrupts;
use crate::runtime_db::interrupt_live_turn_for_signal_preemption;
use crate::runtime_db::is_agent_turn_in_progress;
use crate::runtime_db::latest_open_task_by_kind;
use crate::runtime_db::list_open_tasks;
use crate::runtime_db::list_queued_tasks;
use crate::runtime_db::list_recent_turn_signals;
use crate::runtime_db::list_task_checkpoints;
use crate::runtime_db::load_active_agent_turn;
use crate::runtime_db::load_active_task;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::load_latest_completed_agent_turn;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::load_proactive_contact_candidate_by_dispatch_task;
use crate::runtime_db::load_task_by_id;
use crate::runtime_db::record_bios_dialogue;
use crate::runtime_db::record_brain_usage_event;
use crate::runtime_db::record_homepage_revision;
use crate::runtime_db::record_memory;
use crate::runtime_db::record_proactive_contact_dispatch_result;
use crate::runtime_db::recover_orphaned_active_turns;
use crate::runtime_db::recover_orphaned_review_waits;
use crate::runtime_db::refresh_task_grosshirn_boost;
use crate::runtime_db::register_loop_incident;
use crate::runtime_db::release_task_grosshirn_boost;
use crate::runtime_db::reprioritize_tasks;
use crate::runtime_db::requeue_task;
use crate::runtime_db::requeue_task_with_checkpoint_kind;
use crate::runtime_db::resolve_loop_incident;
use crate::runtime_db::set_agent_mode;
use crate::runtime_db::start_agent_turn;
use crate::runtime_db::store_learning_entries;
use crate::runtime_db::store_proactive_contact_candidate;
use crate::runtime_db::sync_agent_thread;
use crate::runtime_db::sync_model_resources;
use crate::runtime_db::sync_owner_trust;
use crate::runtime_db::sync_resources_from_census;
use crate::runtime_db::sync_skills;
use crate::runtime_db::task_has_active_grosshirn_boost;
use crate::runtime_db::watchdog_interrupt_live_turn;
use crate::runtime_db::yield_active_task_for_preemption;
use crate::tooling::ExecCommandDirective;
use crate::tooling::apply_homepage_update;
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
use std::collections::BTreeMap;
use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;
use std::process::Command;
use std::time::Duration;
use std::time::Instant;

struct RunningAgenticTask {
    turn_id: i64,
    task_id: i64,
    task_title: String,
    task_kind: String,
    started_at: Instant,
    last_context_progress_event_at: Option<Instant>,
    handle: tokio::task::JoinHandle<()>,
}

#[derive(Debug, Clone)]
struct ContextPreparationProgressSnapshot {
    active_phase: String,
    context_mode: String,
    query_answers: usize,
    phase_completed_loops: usize,
    phase_max_loops: usize,
    total_completed_loops: usize,
    total_max_loops: usize,
    latest_decision: String,
}

struct TaskEscalationDecision {
    task_status: String,
    next_mode: String,
    checkpoint_summary: String,
    checkpoint_detail: String,
    spawn_self_preservation_task: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CompletionReviewResolution {
    Approve,
    Revise,
    Block,
}

fn normalize_completion_review_decision(raw: &str) -> Option<CompletionReviewResolution> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "approve" | "approved" | "accept" | "accepted" | "done" | "complete" => {
            Some(CompletionReviewResolution::Approve)
        }
        "revise" | "rework" | "retry" | "retry_review" | "continue" | "reopen" | "reject" => {
            Some(CompletionReviewResolution::Revise)
        }
        "blocked" | "block" | "hard_blocked" | "hard-blocked" => {
            Some(CompletionReviewResolution::Block)
        }
        _ => None,
    }
}

fn resolve_completion_review(
    task_status: &str,
    review: Option<&CompletionReviewDirective>,
) -> (CompletionReviewResolution, Option<String>) {
    if let Some(review) = review
        && let Some(decision) = normalize_completion_review_decision(&review.decision)
    {
        return (decision, None);
    }

    let note = match task_status {
        "done" => Some(
            "Completion review did not explicitly approve the parent task, so the task stays open."
                .to_string(),
        ),
        "blocked" => Some(
            "The review step failed or could not verify completion cleanly, so the parent task stays open."
                .to_string(),
        ),
        _ => None,
    };
    (CompletionReviewResolution::Revise, note)
}

fn context_preparation_model_post_timeout_secs() -> u64 {
    let base_secs = std::env::var("CTO_AGENT_MODEL_POST_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|secs| *secs >= 1)
        .unwrap_or(180);
    std::env::var("CTO_AGENT_CONTEXT_PREPARATION_MODEL_POST_TIMEOUT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|secs| *secs >= base_secs.max(1))
        .unwrap_or_else(|| base_secs.saturating_mul(4).max(1200))
}

fn context_preparation_progress_interval() -> Duration {
    Duration::from_secs(
        std::env::var("CTO_AGENT_CONTEXT_PREPARATION_PROGRESS_EVENT_INTERVAL_SECS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|secs| *secs >= 15)
            .unwrap_or(60),
    )
}

fn context_preparation_total_max_loops(paths: &Paths) -> usize {
    load_context_optimization_policy(paths)
        .max_total_loops
        .max(1)
}

fn context_preparation_stale_after_secs(base_secs: u64) -> u64 {
    let min_secs = context_preparation_model_post_timeout_secs().saturating_add(180);
    std::env::var("CTO_AGENT_CONTEXT_PREPARATION_STALE_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|secs| *secs >= base_secs.max(min_secs))
        .unwrap_or_else(|| base_secs.saturating_mul(4).max(base_secs).max(min_secs))
}

fn kleinhirn_repair_retry_cooldown() -> Duration {
    Duration::from_secs(
        std::env::var("CTO_AGENT_KLEINHIRN_REPAIR_RETRY_SECS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|secs| *secs >= 1)
            .unwrap_or(60),
    )
}

fn effective_live_turn_stale_after_secs(base_secs: u64, task_kind: &str) -> u64 {
    if task_kind == "context_preparation" {
        context_preparation_stale_after_secs(base_secs)
    } else {
        base_secs
    }
}

#[derive(Debug)]
enum ContextPreparationRouting {
    Proceed,
    YieldToExisting {
        prep_task_id: i64,
        prep_title: String,
    },
    EnqueueFresh {
        title: String,
        detail: String,
        priority_score: i64,
    },
}

pub fn spawn_supervisor(paths: Paths, started_at: Instant) {
    tokio::spawn(async move {
        let tick_ms = std::env::var("CTO_AGENT_SUPERVISOR_TICK_MS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|value| *value >= 100)
            .unwrap_or(1000);
        let mut interval = tokio::time::interval(Duration::from_millis(tick_ms));
        let mut ticks: u64 = 0;
        let mut agentic_task: Option<RunningAgenticTask> = None;
        let mut kleinhirn_repair_task: Option<tokio::task::JoinHandle<()>> = None;
        let mut last_kleinhirn_repair_started_at: Option<Instant> = None;
        let mut email_sync_task: Option<tokio::task::JoinHandle<anyhow::Result<MailSyncOutcome>>> =
            None;
        let mut last_email_sync_started_at: Option<Instant> = None;
        let mut last_review_recovery_started_at: Option<Instant> = None;
        loop {
            interval.tick().await;
            ticks += 1;
            let stale_after_secs = std::env::var("CTO_AGENT_ACTIVE_TURN_STALE_SECS")
                .ok()
                .and_then(|raw| raw.parse::<u64>().ok())
                .unwrap_or(300);
            if let Some(running) = agentic_task.take() {
                if running.handle.is_finished() {
                    match running.handle.await {
                        Ok(()) => {}
                        Err(err) => {
                            let summary =
                                format!("Turn {} ist im Rust-Kern abgestuerzt.", running.turn_id);
                            let detail = err.to_string();
                            let mut recovered_review_parent: Option<(i64, String)> = None;
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
                            if matches!(running.task_kind.as_str(), "self_review" | "worker_review")
                            {
                                if let Ok(Some(review_task)) =
                                    load_task_by_id(&paths, running.task_id)
                                {
                                    let failure_detail = format!(
                                        "{}\n\nThe review task itself crashed inside the Rust kernel, so the parent task was reopened instead of being left in await_review.",
                                        detail
                                    );
                                    if complete_review_task(
                                        &paths,
                                        &review_task,
                                        "continue",
                                        &summary,
                                        &failure_detail,
                                        None,
                                    )
                                    .is_ok()
                                    {
                                        if let Some(parent_task_id) = review_task.parent_task_id {
                                            recovered_review_parent =
                                                Some((parent_task_id, review_task.title.clone()));
                                        }
                                    }
                                } else {
                                    let _ = block_task(
                                        &paths,
                                        running.task_id,
                                        &summary,
                                        &detail,
                                        None,
                                    );
                                }
                            } else {
                                let _ =
                                    block_task(&paths, running.task_id, &summary, &detail, None);
                            }
                            if let Some((parent_task_id, review_title)) = recovered_review_parent {
                                let recovered_summary = format!(
                                    "Turn {} crashed during review, but reopened parent task {}.",
                                    running.turn_id, parent_task_id
                                );
                                let recovered_detail = format!(
                                    "{}\n\nThe crashed review task did not block the workstream. Its parent task was reopened and can continue.",
                                    detail
                                );
                                let _ = complete_agent_turn(
                                    &paths,
                                    running.turn_id,
                                    "crashed_recovered",
                                    "review_continue",
                                    &recovered_summary,
                                    &recovered_detail,
                                );
                                if let Ok(Some(parent_task)) =
                                    load_task_by_id(&paths, parent_task_id)
                                {
                                    let _ = set_agent_mode(
                                        &paths,
                                        "execute_task",
                                        Some(parent_task_id),
                                        &parent_task.title,
                                        "A crashed review task was recovered by reopening the parent task.",
                                    );
                                } else {
                                    let _ = set_agent_mode(
                                        &paths,
                                        "reprioritize",
                                        Some(parent_task_id),
                                        &review_title,
                                        "A crashed review task reopened its parent and returned the loop to work selection.",
                                    );
                                }
                                eprintln!(
                                    "bounded review turn crashed for task {} / turn {} but parent {} was reopened",
                                    running.task_id, running.turn_id, parent_task_id
                                );
                                continue;
                            }
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
                    let mut running = running;
                    if let Ok(Some(active_task)) = load_task_by_id(&paths, running.task_id) {
                        match find_interrupt_preemption_candidate(
                            &paths,
                            &active_task,
                            running.turn_id,
                        ) {
                            Ok(Some((queued_task, signal))) => {
                                let age = running.started_at.elapsed().as_secs();
                                if let Err(err) = interrupt_live_turn_for_signal_preemption(
                                    &paths,
                                    running.turn_id,
                                    running.task_id,
                                    &running.task_title,
                                    age,
                                    queued_task.id,
                                    &queued_task.title,
                                    &signal,
                                ) {
                                    eprintln!(
                                        "interrupt preemption failed for task {} / turn {}: {}",
                                        running.task_id, running.turn_id, err
                                    );
                                    agentic_task = Some(running);
                                } else {
                                    eprintln!(
                                        "interrupt preempted live turn {} for task {} in favor of queued task {}",
                                        running.turn_id, running.task_id, queued_task.id
                                    );
                                }
                                continue;
                            }
                            Ok(None) => {}
                            Err(err) => eprintln!(
                                "interrupt preemption check failed for task {} / turn {}: {}",
                                running.task_id, running.turn_id, err
                            ),
                        }
                    }
                    let age = running.started_at.elapsed().as_secs();
                    let effective_stale_after_secs =
                        effective_live_turn_stale_after_secs(stale_after_secs, &running.task_kind);
                    if age > effective_stale_after_secs {
                        if let Err(err) = watchdog_interrupt_live_turn(
                            &paths,
                            running.turn_id,
                            running.task_id,
                            &running.task_title,
                            age,
                            effective_stale_after_secs,
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
                        if running.task_kind == "context_preparation" {
                            let should_emit_progress = running
                                .last_context_progress_event_at
                                .map(|last| {
                                    last.elapsed() >= context_preparation_progress_interval()
                                })
                                .unwrap_or(true);
                            if should_emit_progress {
                                record_context_preparation_active_event(
                                    &paths,
                                    running.task_id,
                                    &running.task_title,
                                    running.turn_id,
                                    age,
                                );
                                running.last_context_progress_event_at = Some(Instant::now());
                            }
                        }
                        agentic_task = Some(running);
                    }
                }
            }

            if let Some(repair_task) = kleinhirn_repair_task.take() {
                if repair_task.is_finished() {
                    if let Err(err) = repair_task.await {
                        eprintln!("kleinhirn repair task crashed: {err}");
                    }
                } else {
                    kleinhirn_repair_task = Some(repair_task);
                }
            }

            if let Some(sync_task) = email_sync_task.take() {
                if sync_task.is_finished() {
                    match sync_task.await {
                        Ok(Ok(outcome)) => {
                            let _ = resolve_loop_incident(
                                &paths,
                                "email_sync_unavailable",
                                &outcome.note,
                            );
                            if outcome.stored_count > 0 {
                                let _ = record_memory(
                                    &paths,
                                    "communication",
                                    "Inbound email sync stored new messages",
                                    &outcome.note,
                                    "email_sync_interrupt_bridge",
                                );
                            }
                        }
                        Ok(Err(err)) => {
                            let detail = err.to_string();
                            let _ = register_loop_incident(
                                &paths,
                                "email_sync_unavailable",
                                "warning",
                                "Periodic inbox sync failed.",
                                &detail,
                                None,
                                None,
                                true,
                                false,
                            );
                            eprintln!("periodic email sync failed: {detail}");
                        }
                        Err(err) => eprintln!("email sync task crashed: {err}"),
                    }
                } else {
                    email_sync_task = Some(sync_task);
                }
            }

            let has_running_turn = agentic_task.is_some();
            let has_active_workstream =
                has_running_turn || load_active_task(&paths).ok().flatten().is_some();
            if should_enter_observe_mode(has_running_turn, has_active_workstream) {
                let _ = set_agent_mode(
                    &paths,
                    "observe",
                    None,
                    "",
                    "Observe current signals, resources and queued work before reprioritization.",
                );
            }

            if let Err(err) = recover_orphaned_active_turns(
                &paths,
                agentic_task.as_ref().map(|running| running.turn_id),
                stale_after_secs,
            ) {
                eprintln!("orphaned turn recovery failed: {err}");
            }
            if should_start_review_wait_recovery(
                last_review_recovery_started_at,
                review_wait_recovery_interval(),
            ) {
                match recover_orphaned_review_waits(&paths) {
                    Ok(recovered) => {
                        if !recovered.is_empty() {
                            eprintln!("reopened orphaned await_review parent tasks: {recovered:?}");
                        }
                    }
                    Err(err) => eprintln!("orphaned review wait recovery failed: {err}"),
                }
                last_review_recovery_started_at = Some(Instant::now());
            }

            match expire_stale_grosshirn_boost(&paths) {
                Ok(Some(note)) => {
                    let _ = record_memory(
                        &paths,
                        "brain_routing",
                        "Temporary grosshirn boost expired",
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

            if should_run_reprioritize(has_active_workstream)
                && let Err(err) = reprioritize_tasks(&paths)
            {
                eprintln!("task reprioritization failed: {err}");
            }
            if let Err(err) = ensure_cto_obligations(&paths) {
                eprintln!("obligation generation failed: {err}");
            }

            if should_start_email_sync_task(
                email_sync_task.is_some(),
                last_email_sync_started_at,
                email_sync_poll_interval(),
            ) {
                let sync_paths = paths.clone();
                email_sync_task = Some(tokio::task::spawn_blocking(move || {
                    run_periodic_email_interrupt_sync(&sync_paths)
                }));
                last_email_sync_started_at = Some(Instant::now());
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
                    if should_start_kleinhirn_repair_task(
                        kleinhirn_repair_task.is_some(),
                        last_kleinhirn_repair_started_at,
                        kleinhirn_repair_retry_cooldown(),
                    ) {
                        let repair_paths = paths.clone();
                        kleinhirn_repair_task = Some(tokio::task::spawn_blocking(move || {
                            match attempt_kleinhirn_runtime_repair(&repair_paths) {
                                Ok(repair_note) => {
                                    let _ = resolve_loop_incident(
                                        &repair_paths,
                                        "kleinhirn_unavailable",
                                        &repair_note,
                                    );
                                    let _ = record_memory(
                                        &repair_paths,
                                        "self_preservation",
                                        "Kernel self-repair restored kleinhirn",
                                        &repair_note,
                                        "kleinhirn_runtime_self_repair",
                                    );
                                    let _ = set_agent_mode(
                                        &repair_paths,
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
                                    let _ = record_memory(
                                        &repair_paths,
                                        "self_preservation",
                                        "Kernel self-repair could not restore kleinhirn",
                                        &repair_err.to_string(),
                                        "kleinhirn_runtime_self_repair_failed",
                                    );
                                }
                            }
                        }));
                        last_kleinhirn_repair_started_at = Some(Instant::now());
                        let _ = set_agent_mode(
                            &paths,
                            "self_preservation",
                            None,
                            "",
                            "Periodic kleinhirn probe failed; runtime repair continues in the background while the supervisor loop stays live.",
                        );
                    }
                    eprintln!("kleinhirn health probe failed: {detail}");
                }
            }

            match advance_browser_subworkers(&paths, 2) {
                Ok(completed) if !completed.is_empty() => {
                    let note = format!(
                        "{} delegated worker jobs returned review work.",
                        completed.len()
                    );
                    let _ = set_agent_mode(&paths, "await_review", None, "", &note);
                }
                Ok(_) => {}
                Err(err) => eprintln!("worker job advance failed: {err}"),
            }

            let should_tick_agentic = should_tick_agentic_loop(ticks, has_active_workstream);
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
                match classify_context_preparation_need(&paths, &task) {
                    Ok(ContextPreparationRouting::Proceed) => {}
                    Ok(ContextPreparationRouting::YieldToExisting {
                        prep_task_id,
                        prep_title,
                    }) => {
                        let summary = format!(
                            "Yielding task {} to existing context preparation task #{}.",
                            task.id, prep_task_id
                        );
                        let detail = format!(
                            "Task #{task_id} {task_title} was selected, but a fresher context-preparation task is already open.\n\nPreparation task #{prep_task_id}: {prep_title}\n\nThe parent task was requeued so the preparation artifact can land first.",
                            task_id = task.id,
                            task_title = task.title,
                            prep_task_id = prep_task_id,
                            prep_title = prep_title,
                        );
                        let _ = requeue_task_with_checkpoint_kind(
                            &paths,
                            task.id,
                            "context_preparation_wait",
                            &summary,
                            &detail,
                            None,
                        );
                        continue;
                    }
                    Ok(ContextPreparationRouting::EnqueueFresh {
                        title,
                        detail,
                        priority_score,
                    }) => {
                        match crate::runtime_db::enqueue_internal_task(
                            &paths,
                            Some(task.id),
                            "context_preparation",
                            &title,
                            &detail,
                            priority_score,
                        ) {
                            Ok(prep_task) => {
                                let summary = format!(
                                    "Queued context preparation task #{} before direct work.",
                                    prep_task.id
                                );
                                let wait_detail = format!(
                                    "Task #{task_id} {task_title} was routed through explicit context preparation so the next direct work step starts from a richer package.\n\nPreparation task #{prep_id}: {prep_title}\nPriority: {priority}",
                                    task_id = task.id,
                                    task_title = task.title,
                                    prep_id = prep_task.id,
                                    prep_title = prep_task.title,
                                    priority = prep_task.priority_score,
                                );
                                let _ = requeue_task_with_checkpoint_kind(
                                    &paths,
                                    task.id,
                                    "context_preparation_wait",
                                    &summary,
                                    &wait_detail,
                                    None,
                                );
                                continue;
                            }
                            Err(err) => {
                                eprintln!(
                                    "failed to enqueue context preparation for task {}: {}",
                                    task.id, err
                                );
                            }
                        }
                    }
                    Err(err) => {
                        eprintln!(
                            "context-preparation classification failed for task {}: {}",
                            task.id, err
                        );
                    }
                }
                if task.task_kind == "local_model_switch" {
                    if let Ok(Some(newer_task)) =
                        latest_open_task_by_kind(&paths, "local_model_switch")
                        && newer_task.id > task.id
                    {
                        let summary = format!(
                            "Local model switch was superseded by newer owner or BIOS request #{}.",
                            newer_task.id
                        );
                        let detail = format!(
                            "Task #{} was supposed to apply {}, but task #{} now requests a newer local model switch.\n\nOld detail:\n{}\n\nNew detail:\n{}",
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
                            "A newer local model switch superseded an older switch.",
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
                                "An explicit owner/BIOS model switch was executed at the kernel layer.\nRequested target: {}\nChanged: {}\nRestarted: {}\nPrevious runtime: {}\nCurrent runtime: {}\nSummary: {}",
                                requested_target.as_deref().unwrap_or("auto"),
                                outcome.changed,
                                outcome.restarted,
                                outcome
                                    .previous_runtime_model
                                    .as_deref()
                                    .unwrap_or("unknown"),
                                outcome
                                    .current_runtime_model
                                    .as_deref()
                                    .unwrap_or("unknown"),
                                outcome.summary,
                            );
                            let _ = record_memory(
                                &paths,
                                "brain_runtime",
                                "Local kleinhirn model switch executed",
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
                                "Local kleinhirn model switch failed.",
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
                                "Grosshirn activation prepared",
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
                    &format!("Task {} entered bounded {} mode.", task.id, execution_mode),
                );
                if task.task_kind == "proactive_contact_dispatch" {
                    if let Err(err) = execute_proactive_contact_dispatch_task(&paths, &task) {
                        let summary =
                            format!("Dispatch task {} could not be completed cleanly.", task.id);
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
                match decide_context_package_preflight(&paths, &task) {
                    Ok(ContextPackagePreflightDecision::ReuseLatest) => {
                        let _ = crate::runtime_db::record_resource_status(
                            &paths,
                            "agentic_loop",
                            "context_preflight",
                            "reuse_latest",
                            &format!(
                                "Task #{} {} reused the latest prepared context package because no new interrupt was recorded and the estimated active context remains below the compaction threshold.",
                                task.id, task.title
                            ),
                        );
                    }
                    Ok(ContextPackagePreflightDecision::Refresh(trigger)) => {
                        let refresh_result = match trigger {
                            crate::context_controller::ContextCompactionTrigger::Auto => {
                                prepare_context_package(&paths, &task)
                            }
                            crate::context_controller::ContextCompactionTrigger::Interrupt => {
                                prepare_context_package_with_trigger(&paths, &task, trigger)
                            }
                        };
                        if let Err(err) = refresh_result {
                            let summary =
                                format!("Task {} could not build a context package.", task.id);
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
                    }
                    Err(err) => {
                        let summary = format!("Task {} could not decide context preflight.", task.id);
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
                        eprintln!("context preflight decision failed for task {}: {err}", task.id);
                        continue;
                    }
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
                        let summary = format!("Task {} could not start a turn.", task.id);
                        let detail = err.to_string();
                        let _ = block_task(&paths, task.id, &summary, &detail, None);
                        eprintln!("failed to start turn for task {}: {err}", task.id);
                        continue;
                    }
                };
                if task.task_kind == "context_preparation" {
                    record_context_preparation_started_event(&paths, &task, turn.id);
                }
                let loop_paths = paths.clone();
                let task_id = task.id;
                let turn_id = turn.id;
                let task_title = task.title.clone();
                let task_kind = task.task_kind.clone();
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
                                .unwrap_or_else(|| {
                                    "Task run finished without summary.".to_string()
                                });
                            let mut output_text = result
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
                            let mut next_mode =
                                result.next_mode.as_deref().unwrap_or(inferred_next_mode);
                            let mut next_mode =
                                normalize_sticky_continue_mode(&task, &task_status, next_mode);
                            if looks_like_invalid_structured_artifact(&checkpoint_summary)
                                || looks_like_invalid_structured_artifact(&output_text)
                            {
                                let artifact =
                                    if looks_like_invalid_structured_artifact(&output_text) {
                                        output_text.trim().to_string()
                                    } else {
                                        checkpoint_summary.trim().to_string()
                                    };
                                checkpoint_summary =
                                    "The model returned incomplete JSON; bounded retry instead of false success."
                                        .to_string();
                                checkpoint_detail = if artifact.is_empty() {
                                    checkpoint_summary.clone()
                                } else {
                                    format!(
                                        "{checkpoint_summary}\n\nRohes Modellartefakt:\n{artifact}"
                                    )
                                };
                                task_status = "continue".to_string();
                                next_mode = normalize_sticky_continue_mode(
                                    &task,
                                    &task_status,
                                    "reprioritize",
                                );
                            }
                            if task_used_grosshirn {
                                let _ = refresh_task_grosshirn_boost(&loop_paths, task.id);
                            }
                            if let Some(usage) = result.model_usage.as_ref() {
                                let brain_tier = if task_used_grosshirn {
                                    "grosshirn"
                                } else {
                                    "kleinhirn"
                                };
                                let source_label = if task_used_grosshirn {
                                    "external grosshirn"
                                } else {
                                    "local kleinhirn"
                                };
                                let note = if task_used_grosshirn {
                                    format!(
                                        "Task #{} {} used external grosshirn for a bounded step.",
                                        task.id, task.title
                                    )
                                } else {
                                    format!(
                                        "Task #{} {} used the local active model for a bounded step.",
                                        task.id, task.title
                                    )
                                };
                                let _ = record_brain_usage_event(
                                    &loop_paths,
                                    Some(task.id),
                                    Some(turn_id),
                                    brain_tier,
                                    source_label,
                                    result.model.as_deref().unwrap_or("unknown"),
                                    usage.input_tokens,
                                    usage.output_tokens,
                                    usage.total_tokens,
                                    usage.duration_ms,
                                    usage.estimated_cost_usd,
                                    &note,
                                );
                            }
                            if task_fell_back_to_kleinhirn {
                                let detail = "Grosshirn execution fell back to the local kleinhirn; the temporary boost was closed for this task.";
                                let _ = release_task_grosshirn_boost(&loop_paths, task.id, detail);
                                checkpoint_summary = format!(
                                    "{} Grosshirn failed; the local kleinhirn fallback took over.",
                                    checkpoint_summary
                                );
                                checkpoint_detail = format!("{checkpoint_detail}\n\n{detail}");
                            }
                            let loop_safety = load_loop_safety_policy(&loop_paths);
                            let self_preservation = load_self_preservation_state(&loop_paths);
                            if let Some(escalation) = assess_task_stuck_risk(
                                &loop_paths,
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
                                        "Recalibrate self-preservation for the running task",
                                        &format!(
                                            "Analyze why task #{} {} is not advancing productively right now, and decide on resource requests, delegation, context recutting, or later resumption.\n\n{}",
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
                                        checkpoint_summary =
                                            format!("{} {}", checkpoint_summary, outcome.summary);
                                        checkpoint_detail = format!(
                                            "{}\n\nKernel self-repair:\nThe local kleinhirn upgrade was executed directly after repeated stalling in the local review path.\nChanged: {}\nRestarted: {}\nPrevious runtime: {}\nCurrent runtime: {}",
                                            checkpoint_detail,
                                            outcome.changed,
                                            outcome.restarted,
                                            outcome
                                                .previous_runtime_model
                                                .as_deref()
                                                .unwrap_or("unknown"),
                                            outcome
                                                .current_runtime_model
                                                .as_deref()
                                                .unwrap_or("unknown"),
                                        );
                                        let _ = record_memory(
                                            &loop_paths,
                                            "brain_runtime",
                                            "Kernel self-repair executed a local kleinhirn upgrade",
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
                                            "{} Kernel self-repair for the local kleinhirn upgrade failed.",
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
                                            "{} Homepage was extended directly.",
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
                                            "Homepage mutation failed inside the bounded task run.",
                                            &detail,
                                            Some(task.id),
                                            Some(turn_id),
                                            false,
                                            false,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Homepage mutation failed.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nHomepage tool error: {}",
                                            checkpoint_detail, detail
                                        );
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
                                        let prior_summary = checkpoint_summary.clone();
                                        let prior_detail = checkpoint_detail.clone();
                                        checkpoint_summary = session_summary;
                                        checkpoint_detail = format!(
                                            "{}\n\nModel-declared checkpoint before exec-session grounding:\n{}\n\nExec session result:\n{}",
                                            prior_detail, prior_summary, session_detail
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode =
                                            continue_next_mode_for_task(&task).to_string();
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
                                        let prior_summary = checkpoint_summary.clone();
                                        let prior_detail = checkpoint_detail.clone();
                                        checkpoint_summary =
                                            "Exec-session action failed.".to_string();
                                        checkpoint_detail = format!(
                                            "{}\n\nModel-declared checkpoint before exec-session grounding:\n{}\n\nExec session error: {}",
                                            prior_detail, prior_summary, detail
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode =
                                            continue_next_mode_for_task(&task).to_string();
                                    }
                                }
                            } else if let Some(exec_directive) = result.exec_directive.as_ref() {
                                match run_bounded_command(&loop_paths, &task, exec_directive) {
                                    Ok(exec_result) => {
                                        let justification = exec_directive
                                            .justification
                                            .as_deref()
                                            .unwrap_or("none");
                                        let prior_summary = checkpoint_summary.clone();
                                        let prior_detail = checkpoint_detail.clone();
                                        checkpoint_summary = grounded_exec_command_summary(
                                            &exec_directive.command,
                                            exec_result.exit_code,
                                            exec_result.timed_out,
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nModel-declared checkpoint before machine grounding:\n{}\n\nBounded exec result:\nCommand: {:?}\nWorkdir: {}\nJustification: {}\nStatus: {}\nExit code: {:?}\nTimed out: {}\nSTDOUT:\n{}\n\nSTDERR:\n{}",
                                            prior_detail,
                                            prior_summary,
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
                                        normalized_next_mode =
                                            continue_next_mode_for_task(&task).to_string();
                                        if let Some((mail_summary, mail_detail)) =
                                            visible_owner_mail_send_completion(
                                                &loop_paths,
                                                &task,
                                                exec_directive,
                                                &exec_result,
                                            )
                                        {
                                            checkpoint_summary = mail_summary;
                                            checkpoint_detail = format!(
                                                "{}\n\nOwner mail completion:\n{}",
                                                checkpoint_detail, mail_detail
                                            );
                                            task_status = "done".to_string();
                                            normalized_next_mode = "review".to_string();
                                        }
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
                                        let prior_summary = checkpoint_summary.clone();
                                        let prior_detail = checkpoint_detail.clone();
                                        checkpoint_summary =
                                            "Bounded command-exec failed.".to_string();
                                        checkpoint_detail = format!(
                                            "{}\n\nModel-declared checkpoint before machine grounding:\n{}\n\nCommand exec error: {}",
                                            prior_detail, prior_summary, detail
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode =
                                            continue_next_mode_for_task(&task).to_string();
                                    }
                                }
                            } else if let Some(browser_directive) =
                                result.browser_directive.as_ref()
                            {
                                match run_browser_action(&loop_paths, &task, browser_directive) {
                                    Ok(browser_result) => {
                                        checkpoint_summary = format!(
                                            "{} Browser engine action executed: {}",
                                            checkpoint_summary, browser_directive.action
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBrowser action result:\nAction: {}\nURL: {}\nStatus: {}\nBrowser status: {}\nArtifact: {}\nSTDOUT:\n{}\n\nSTDERR:\n{}",
                                            checkpoint_detail,
                                            browser_directive.action,
                                            browser_directive.url.as_deref().unwrap_or(""),
                                            browser_result.status,
                                            browser_result.browser_status,
                                            browser_result
                                                .artifact_path
                                                .as_deref()
                                                .unwrap_or("none"),
                                            browser_result.stdout,
                                            browser_result.stderr,
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode =
                                            continue_next_mode_for_task(&task).to_string();
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
                                            "{} Browser engine action failed.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBrowser engine error: {}",
                                            checkpoint_detail, detail
                                        );
                                        if task_status != "blocked" {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode =
                                            continue_next_mode_for_task(&task).to_string();
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
                                    context_directive.concern.as_deref().unwrap_or("none"),
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

                                let needs_history_research = normalized_next_mode
                                    == "historical_research"
                                    || matches!(
                                        context_directive.action.as_str(),
                                        "expand_history" | "question_compaction"
                                    )
                                    || context_directive.history_research_query.is_some();
                                let history_should_stay_inline =
                                    should_keep_history_research_inline(
                                        &loop_paths,
                                        &task,
                                        needs_history_research,
                                        &checkpoint_detail,
                                    );
                                if history_should_stay_inline {
                                    checkpoint_summary = format!(
                                        "{} Local workspace evidence was already gathered in this bounded turn, so historical reload stays inline on the same task instead of branching into a separate research detour.",
                                        checkpoint_summary
                                    );
                                    checkpoint_detail = format!(
                                        "{}\n\nWorkstream continuity note: this task keeps historical follow-up inline on the same workstream instead of spawning a separate historical_research task.",
                                        checkpoint_detail
                                    );
                                    if normalized_next_mode == "historical_research"
                                        || needs_history_research
                                    {
                                        normalized_next_mode =
                                            continue_next_mode_for_task(&task).to_string();
                                    }
                                } else if should_enqueue_history_research(
                                    &task.task_kind,
                                    needs_history_research,
                                ) {
                                    let research_detail = format!(
                                        "Perform targeted historical reload for task #{} {}.\n\nAgentic context reason: {}\nContext concern: {}\nTargeted query: {}\n\nUse raw history, checkpoints, BIOS dialogue, and memory selectively. The goal is not blind expansion, but a precise reload step for the next bounded run.",
                                        task.id,
                                        task.title,
                                        context_directive.action,
                                        context_directive.concern.as_deref().unwrap_or("none"),
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
                                        &format!("Prepare historical reload for task {}", task.id),
                                        &research_detail,
                                        research_priority,
                                    );
                                    if normalized_next_mode == "historical_research" {
                                        normalized_next_mode = "reprioritize".to_string();
                                    }
                                } else if needs_history_research
                                    && task.task_kind == "context_preparation"
                                {
                                    checkpoint_summary = format!(
                                        "{} Context preparation kept missing evidence inside the preparation loop instead of spawning historical research.",
                                        checkpoint_summary
                                    );
                                    checkpoint_detail = format!(
                                        "{}\n\nContext preparation requested historical reload, but the kernel kept the work inside the preparation loop. Missing evidence should flow through preparedContextArtifact.questions and review.missingEvidence instead of enqueuing a separate historical_research task.",
                                        checkpoint_detail
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
                                                "{} System census including mistralrs tune was executed.",
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
                                                "{} System census failed.",
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
                                let brain_target_task =
                                    load_task_by_id(&loop_paths, brain_target_task_id)
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
                                                "Agentic decision: temporary grosshirn is justified for this task.",
                                            ),
                                            1120,
                                        );
                                        checkpoint_summary = format!(
                                            "{} Temporary grosshirn boost was activated for task #{}.",
                                            checkpoint_summary, brain_target_task_id
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBrain action:\nAction: {}\nTarget model hint: {}\nNote: {}\nApplied to task: #{} {}",
                                            checkpoint_detail,
                                            brain_directive.action,
                                            brain_directive
                                                .target_model
                                                .as_deref()
                                                .unwrap_or("not supplied"),
                                            brain_directive.note.as_deref().unwrap_or("none"),
                                            brain_target_task_id,
                                            brain_target_title,
                                        );
                                        if task.task_kind == "grosshirn_activation"
                                            && (task_used_grosshirn || task_fell_back_to_kleinhirn)
                                        {
                                            task_status = "done".to_string();
                                        } else if task_status != "blocked" && task_status != "done"
                                        {
                                            task_status = "continue".to_string();
                                        }
                                        normalized_next_mode = "reprioritize".to_string();
                                    } else {
                                        checkpoint_summary = format!(
                                            "{} A temporary grosshirn boost was requested, but is not available yet.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nBrain action:\nAction: {}\nTarget model hint: {}\nNote: {}\nResult: grosshirn is not available or not enabled yet.",
                                            checkpoint_detail,
                                            brain_directive.action,
                                            brain_directive
                                                .target_model
                                                .as_deref()
                                                .unwrap_or("not supplied"),
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
                                            "Agentic decision: an external grosshirn boost is no longer needed for this task.",
                                        ),
                                    );
                                    checkpoint_summary = format!(
                                        "{} Temporary grosshirn boost was released for task #{}.",
                                        checkpoint_summary, brain_target_task_id
                                    );
                                    checkpoint_detail = format!(
                                        "{}\n\nBrain action:\nAction: {}\nTarget model hint: {}\nNote: {}\nReleased from task: #{} {}",
                                        checkpoint_detail,
                                        brain_directive.action,
                                        brain_directive
                                            .target_model
                                            .as_deref()
                                            .unwrap_or("not supplied"),
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
                                                brain_directive
                                                    .target_model
                                                    .as_deref()
                                                    .unwrap_or("not supplied"),
                                                brain_directive.note.as_deref().unwrap_or("none"),
                                                outcome.changed,
                                                outcome.restarted,
                                                outcome
                                                    .previous_runtime_model
                                                    .as_deref()
                                                    .unwrap_or("unknown"),
                                                outcome
                                                    .current_runtime_model
                                                    .as_deref()
                                                    .unwrap_or("unknown"),
                                            );
                                            let _ = record_memory(
                                                &loop_paths,
                                                "brain_runtime",
                                                "Vision-capable local kleinhirn switched",
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
                                                "{} Browser/vision kleinhirn upgrade failed.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nLocal browser-vision kleinhirn runtime action failed:\nAction: {}\nTarget model hint: {}\nNote: {}\nError: {}",
                                                checkpoint_detail,
                                                brain_directive.action,
                                                brain_directive
                                                    .target_model
                                                    .as_deref()
                                                    .unwrap_or("not supplied"),
                                                brain_directive.note.as_deref().unwrap_or("none"),
                                                detail,
                                            );
                                            if task_status != "blocked" {
                                                task_status = "continue".to_string();
                                            }
                                            normalized_next_mode = "reprioritize".to_string();
                                        }
                                    }
                                } else if brain_directive
                                    .action
                                    .eq_ignore_ascii_case("upgrade_local_kleinhirn")
                                {
                                    let requested_target =
                                        brain_directive.target_model.clone().or_else(|| {
                                            extract_requested_local_kleinhirn_model(&task.detail)
                                        });
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
                                                requested_target
                                                    .as_deref()
                                                    .unwrap_or("not supplied"),
                                                brain_directive.note.as_deref().unwrap_or("none"),
                                                outcome.changed,
                                                outcome.restarted,
                                                outcome
                                                    .previous_runtime_model
                                                    .as_deref()
                                                    .unwrap_or("unknown"),
                                                outcome
                                                    .current_runtime_model
                                                    .as_deref()
                                                    .unwrap_or("unknown"),
                                            );
                                            let _ = record_memory(
                                                &loop_paths,
                                                "brain_runtime",
                                                "Local kleinhirn switched",
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
                                                "{} Local kleinhirn upgrade failed.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nLocal kleinhirn runtime action failed:\nAction: {}\nTarget model hint: {}\nNote: {}\nError: {}",
                                                checkpoint_detail,
                                                brain_directive.action,
                                                brain_directive
                                                    .target_model
                                                    .as_deref()
                                                    .unwrap_or("not supplied"),
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
                                let grosshirn_candidates = if policy.grosshirn_candidates.is_empty()
                                {
                                    "none configured".to_string()
                                } else {
                                    policy
                                        .grosshirn_candidates
                                        .iter()
                                        .map(|candidate| {
                                            format!(
                                                "{} ({})",
                                                candidate.official_label, candidate.model_id
                                            )
                                        })
                                        .collect::<Vec<_>>()
                                        .join(", ")
                                };
                                if task.task_kind == "model_or_resource"
                                    && local_kleinhirn_available
                                {
                                    let local_upgrade_detail = format!(
                                        "Task #{task_id} ({title}) requested `request_resources` even though a stronger local kleinhirn is already available.\n\nReason:\n{checkpoint_detail}\n\nCurrent brain access mode: {brain_mode}\nCurrently recommended local kleinhirn: {kleinhirn}\n\nNext bounded step:\n1. Stay on the local kleinhirn decision path.\n2. Check or execute the recommended local upgrade.\n3. Only if the local upgrade fails honestly or provably is not enough may you expand into grosshirn or resource procurement.",
                                        task_id = task.id,
                                        title = task.title,
                                        checkpoint_detail = checkpoint_detail,
                                        brain_mode = trust.brain_access_mode,
                                        kleinhirn = crate::contracts::describe_kleinhirn_selection(
                                            &policy, &census,
                                        ),
                                    );
                                    let _ = crate::runtime_db::enqueue_internal_task(
                                        &loop_paths,
                                        Some(task.id),
                                        "model_or_resource",
                                        "Actively upgrade the local kleinhirn or reject it with reasons",
                                        &local_upgrade_detail,
                                        875,
                                    );
                                    checkpoint_summary = format!(
                                        "{} Local kleinhirn follow-up work was queued.",
                                        checkpoint_summary
                                    );
                                    task_status = "continue".to_string();
                                    normalized_next_mode = "reprioritize".to_string();
                                } else if grosshirn_available {
                                    let boost_note = "Grosshirn is already enabled and configured; the same task now continues through it instead of new procurement.";
                                    match arm_task_grosshirn_boost(
                                        &loop_paths,
                                        task.id,
                                        &task.title,
                                        boost_note,
                                        1120,
                                    ) {
                                        Ok(_) => {
                                            checkpoint_summary = format!(
                                                "{} Existing grosshirn will be used for the same task instead of new procurement.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{checkpoint_detail}\n\nAlready available grosshirn is being routed directly to task #{task_id} {title} instead of creating more procurement loops.\nBrain access mode: {brain_mode}\nCurrently recommended local kleinhirn: {kleinhirn}\nGrosshirn candidates: {grosshirn_candidates}\nBoost note: {boost_note}",
                                                checkpoint_detail = checkpoint_detail,
                                                task_id = task.id,
                                                title = task.title,
                                                brain_mode = trust.brain_access_mode,
                                                kleinhirn =
                                                    crate::contracts::describe_kleinhirn_selection(
                                                        &policy, &census,
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
                                                "{} Existing grosshirn should have been used instead of procurement, but the boost could not be set.",
                                                checkpoint_summary
                                            );
                                            checkpoint_detail = format!(
                                                "{checkpoint_detail}\n\nGrosshirn boost could not be set for task #{task_id} {title}: {err}",
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
                                        "Task #{task_id} ({title}) requested `request_resources`, but the runtime already contains a grosshirn configuration.\n\nReason:\n{checkpoint_detail}\n\nCurrent brain access mode: {brain_mode}\nCurrently recommended local kleinhirn: {kleinhirn}\nPossible grosshirn candidates: {grosshirn_candidates}\n\nNext bounded step:\n1. First activate the already prepared grosshirn access cleanly in brain access.\n2. Then use the same Infinity Loop with grosshirn plus local fallback.\n3. Do not create a new procurement loop for credentials that already exist.",
                                        task_id = task.id,
                                        title = task.title,
                                        checkpoint_detail = checkpoint_detail,
                                        brain_mode = trust.brain_access_mode,
                                        kleinhirn = crate::contracts::describe_kleinhirn_selection(
                                            &policy, &census,
                                        ),
                                        grosshirn_candidates = grosshirn_candidates,
                                    );
                                    let _ = crate::runtime_db::enqueue_internal_task(
                                        &loop_paths,
                                        Some(task.id),
                                        "grosshirn_activation",
                                        "Activate and verify grosshirn mode",
                                        &activation_detail,
                                        865,
                                    );
                                    checkpoint_summary = format!(
                                        "{} Grosshirn activation instead of new procurement was queued.",
                                        checkpoint_summary
                                    );
                                    if task_status != "blocked" {
                                        task_status = "continue".to_string();
                                    }
                                    normalized_next_mode = "reprioritize".to_string();
                                } else {
                                    let procurement_detail = format!(
                                        "Task #{task_id} ({title}) requested `request_resources`.\n\nReason:\n{checkpoint_detail}\n\nCurrent brain access mode: {brain_mode}\nCurrently recommended local kleinhirn: {kleinhirn}\nPossible grosshirn candidates: {grosshirn_candidates}\n\nNext bounded step:\n1. Check first whether a better local kleinhirn on the same host is possible.\n2. If local upgrade is not enough, prepare a clear owner request for additional resources or grosshirn access.\n3. If the owner already allowed grosshirn access and credentials are configured, then use the same Infinity Loop with grosshirn plus local fallback.",
                                        task_id = task.id,
                                        title = task.title,
                                        checkpoint_detail = checkpoint_detail,
                                        brain_mode = trust.brain_access_mode,
                                        kleinhirn = crate::contracts::describe_kleinhirn_selection(
                                            &policy, &census,
                                        ),
                                        grosshirn_candidates = grosshirn_candidates,
                                    );
                                    let _ = crate::runtime_db::enqueue_internal_task(
                                        &loop_paths,
                                        Some(task.id),
                                        "grosshirn_procurement",
                                        "Prepare resource and grosshirn approval",
                                        &procurement_detail,
                                        860,
                                    );
                                    checkpoint_summary = format!(
                                        "{} Resource/grosshirn follow-up work was queued.",
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
                                            "{} New follow-up task #{} was queued.",
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
                                            "{} Follow-up task could not be queued.",
                                            checkpoint_summary
                                        );
                                        checkpoint_detail = format!(
                                            "{}\n\nFollow-up task enqueue error: {}",
                                            checkpoint_detail, detail
                                        );
                                    }
                                }
                            }
                            let used_machine_path = result.exec_session_directive.is_some()
                                || result.exec_directive.is_some()
                                || result.browser_directive.is_some();
                            if !used_machine_path
                                && workspace_execution_contract_active(&loop_paths, task.id)
                            {
                                let prior_summary = checkpoint_summary.clone();
                                let prior_output_text = output_text.clone();
                                checkpoint_summary = grounded_workspace_non_machine_summary();
                                checkpoint_detail = format!(
                                    "{}\n\nWorkspace execution contract note: no exec/browser machine path ran in this bounded turn, so code/build/test/exec-session progress is not persisted as verified.\nModel-declared checkpoint summary before contract grounding:\n{}",
                                    checkpoint_detail, prior_summary
                                );
                                output_text =
                                    grounded_workspace_non_machine_output_text(&prior_output_text);
                                if task_status == "blocked" {
                                    task_status = "continue".to_string();
                                }
                                if normalized_next_mode == "blocked" {
                                    normalized_next_mode =
                                        execution_mode_for_task(&task).to_string();
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
                                    "{} Learning path could not be updated.",
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
                                    "{} People-path proactivity could not be updated.",
                                    checkpoint_summary
                                );
                                checkpoint_detail = format!(
                                    "{}\n\nProactive contact error: {}",
                                    checkpoint_detail, detail
                                );
                            }
                            if let Some(escalation) = assess_task_stuck_risk(
                                &loop_paths,
                                &task,
                                &loop_safety,
                                &self_preservation.current_stage,
                                &checkpoint_summary,
                                &checkpoint_detail,
                                &task_status,
                                &normalized_next_mode,
                            ) {
                                task_status = escalation.task_status;
                                normalized_next_mode = escalation.next_mode;
                                checkpoint_summary = escalation.checkpoint_summary;
                                checkpoint_detail = escalation.checkpoint_detail;
                            }
                            let task_supports_brain_switch =
                                task_kind_supports_task_local_brain_switch(&task.task_kind);
                            let has_active_task_grosshirn =
                                task_has_active_grosshirn_boost(&loop_paths, task.id);
                            if task_supports_brain_switch
                                && task_status == "continue"
                                && normalized_next_mode == "review"
                                && has_active_task_grosshirn
                            {
                                let park_note = "Temporary grosshirn did not unlock further progress on this task. The task is parked during grosshirn cooldown so the queue can progress elsewhere before this task returns.";
                                let _ =
                                    release_task_grosshirn_boost(&loop_paths, task.id, park_note);
                                checkpoint_summary = format!(
                                    "{} Task parked for grosshirn cooldown before later return.",
                                    checkpoint_summary
                                );
                                checkpoint_detail = format!(
                                    "{}\n\nBrain routing:\n{}",
                                    checkpoint_detail, park_note
                                );
                                task_status = "continue".to_string();
                                normalized_next_mode = "reprioritize".to_string();
                            } else if task_supports_brain_switch
                                && task_status == "continue"
                                && normalized_next_mode == "review"
                                && !has_active_task_grosshirn
                                && !task_used_grosshirn
                                && !task_fell_back_to_kleinhirn
                                && grosshirn_boost_available(&loop_paths)
                            {
                                let activation_note = "Local progress stalled on this task, so the kernel escalated the same task into temporary grosshirn instead of parking it or bouncing immediately into review.";
                                let _ = arm_task_grosshirn_boost(
                                    &loop_paths,
                                    task.id,
                                    &task.title,
                                    activation_note,
                                    task.priority_score.max(1120),
                                );
                                checkpoint_summary = format!(
                                    "{} Temporary grosshirn boost was activated for the same task.",
                                    checkpoint_summary
                                );
                                checkpoint_detail = format!(
                                    "{}\n\nBrain routing:\n{}",
                                    checkpoint_detail, activation_note
                                );
                                task_status = "continue".to_string();
                                normalized_next_mode =
                                    continue_next_mode_for_task(&task).to_string();
                            } else if task_supports_brain_switch
                                && task_status == "continue"
                                && normalized_next_mode != "blocked"
                                && task_used_grosshirn
                                && !task_fell_back_to_kleinhirn
                                && !has_active_task_grosshirn
                                && grosshirn_boost_available(&loop_paths)
                            {
                                let activation_note = "The task just needed grosshirn to move again, so the temporary grosshirn boost remains active for the next bounded step instead of immediately dropping back to kleinhirn.";
                                let _ = arm_task_grosshirn_boost(
                                    &loop_paths,
                                    task.id,
                                    &task.title,
                                    activation_note,
                                    task.priority_score.max(1120),
                                );
                                checkpoint_summary = format!(
                                    "{} Temporary grosshirn boost remains active for the next bounded step.",
                                    checkpoint_summary
                                );
                                checkpoint_detail = format!(
                                    "{}\n\nBrain routing:\n{}",
                                    checkpoint_detail, activation_note
                                );
                                normalized_next_mode =
                                    continue_next_mode_for_task(&task).to_string();
                            }
                            if task.task_kind == "proactive_contact_review"
                                && task_status != "continue"
                                && normalized_next_mode == "review"
                            {
                                normalized_next_mode = "reprioritize".to_string();
                            }
                            let mut task_outcome_persisted = false;

                            if task.task_kind == "context_preparation" {
                                let preparation_artifact =
                                    result.prepared_context_artifact.as_ref();
                                append_prepared_context_artifact_to_detail(
                                    &mut checkpoint_detail,
                                    preparation_artifact,
                                );
                                let preparation_checkpoints =
                                    list_task_checkpoints(&loop_paths, task_id, 16)
                                        .unwrap_or_default();
                                let preparation_snapshot =
                                    context_preparation_progress_snapshot(&loop_paths, task_id);
                                let missing_preparation_artifact =
                                    task_status != "blocked" && preparation_artifact.is_none();
                                let checkpoint_kind = if missing_preparation_artifact {
                                    let active_phase = preparation_snapshot
                                        .as_ref()
                                        .map(|snapshot| snapshot.active_phase.as_str())
                                        .unwrap_or("query_plan");
                                    context_preparation_checkpoint_kind_for_phase(active_phase)
                                } else {
                                    context_preparation_checkpoint_kind(preparation_artifact)
                                };
                                let validation = preparation_artifact
                                    .map(|artifact| {
                                        validate_prepared_context_artifact(
                                            artifact,
                                            &preparation_checkpoints,
                                            &loop_paths,
                                        )
                                    })
                                    .unwrap_or_else(|| {
                                        let mut validation = PreparedContextValidation::default();
                                        if missing_preparation_artifact {
                                            validation.issues.push(
                                                "context preparation must return `preparedContextArtifact` in every phase".to_string(),
                                            );
                                            let phase_completed_loops = preparation_snapshot
                                                .as_ref()
                                                .map(|snapshot| snapshot.phase_completed_loops)
                                                .unwrap_or(0);
                                            let phase_max_loops = preparation_snapshot
                                                .as_ref()
                                                .map(|snapshot| snapshot.phase_max_loops)
                                                .unwrap_or(4);
                                            let current_iteration = phase_completed_loops
                                                .saturating_add(1);
                                            validation.phase_limit_hit =
                                                current_iteration >= phase_max_loops;
                                        }
                                        validation
                                    });
                                let artifact_ready = validation.ready;
                                let artifact_blocked = preparation_artifact
                                    .map(|artifact| {
                                        artifact.review.decision.eq_ignore_ascii_case("blocked")
                                    })
                                    .unwrap_or(false);
                                if missing_preparation_artifact {
                                    checkpoint_summary = format!(
                                        "{} Context optimization contract violation: missing preparedContextArtifact.",
                                        checkpoint_summary
                                    );
                                    checkpoint_detail = format!(
                                        "{}\n\nContext optimization contract violation: this turn did not return `preparedContextArtifact`. Every context-preparation phase must return the machine-readable artifact (`questions` in query_plan, `blocks` in rewrite/review, and `review` in every phase).",
                                        checkpoint_detail
                                    );
                                }
                                if !validation.issues.is_empty() {
                                    checkpoint_detail = format!(
                                        "{}\n\nContext preparation validation issues:\n- {}",
                                        checkpoint_detail,
                                        validation.issues.join("\n- ")
                                    );
                                }
                                if task_status == "blocked"
                                    || artifact_blocked
                                    || validation.phase_limit_hit
                                    || validation.repeated_from_prior
                                {
                                    if let Some(parent_task_id) = task.parent_task_id {
                                        let _ = block_task(
                                            &loop_paths,
                                            parent_task_id,
                                            "Context preparation for the parent task is blocked.",
                                            &checkpoint_detail,
                                            Some(&output_text),
                                        );
                                    }
                                    if let Err(err) = block_task(
                                        &loop_paths,
                                        task_id,
                                        &checkpoint_summary,
                                        &checkpoint_detail,
                                        Some(&output_text),
                                    ) {
                                        eprintln!(
                                            "failed to block context preparation task {task_id}: {err}"
                                        );
                                    }
                                    task_outcome_persisted = true;
                                    task_status = "blocked".to_string();
                                    normalized_next_mode = "blocked".to_string();
                                    record_context_preparation_checkpoint_event(
                                        &loop_paths,
                                        &task,
                                        turn_id,
                                        checkpoint_kind,
                                        preparation_artifact,
                                        &checkpoint_summary,
                                    );
                                } else if artifact_ready {
                                    if let Err(err) = complete_task(
                                        &loop_paths,
                                        task_id,
                                        &checkpoint_summary,
                                        &checkpoint_detail,
                                        Some(&output_text),
                                    ) {
                                        eprintln!(
                                            "failed to complete context preparation task {task_id}: {err}"
                                        );
                                    }
                                    if let Some(parent_task_id) = task.parent_task_id {
                                        let parent_summary = format!(
                                            "Fresh context preparation from task #{} is ready.",
                                            task.id
                                        );
                                        let parent_detail = format!(
                                            "Preparation task #{prep_id} prepared the next direct work step for parent task #{parent_id}.\n\nPreparation summary:\n{summary}\n\nPreparation artifact:\n{detail}",
                                            prep_id = task.id,
                                            parent_id = parent_task_id,
                                            summary = checkpoint_summary,
                                            detail = checkpoint_detail,
                                        );
                                        if let Err(err) = requeue_task_with_checkpoint_kind(
                                            &loop_paths,
                                            parent_task_id,
                                            "context_preparation_ready",
                                            &parent_summary,
                                            &parent_detail,
                                            Some(&output_text),
                                        ) {
                                            eprintln!(
                                                "failed to requeue parent task {} after context preparation {}: {}",
                                                parent_task_id, task_id, err
                                            );
                                        }
                                    }
                                    task_outcome_persisted = true;
                                    task_status = "done".to_string();
                                    normalized_next_mode = "reprioritize".to_string();
                                    record_context_preparation_checkpoint_event(
                                        &loop_paths,
                                        &task,
                                        turn_id,
                                        checkpoint_kind,
                                        preparation_artifact,
                                        &checkpoint_summary,
                                    );
                                } else {
                                    checkpoint_summary = format!(
                                        "{} Context optimization phase: {}.",
                                        checkpoint_summary,
                                        checkpoint_kind.replace('_', " ")
                                    );
                                    if let Some(artifact) = preparation_artifact {
                                        if !artifact.review.weak_blocks.is_empty() {
                                            checkpoint_summary = format!(
                                                "{} Weak blocks: {}.",
                                                checkpoint_summary,
                                                artifact.review.weak_blocks.join(", ")
                                            );
                                        }
                                    }
                                    if let Err(err) = requeue_task_with_checkpoint_kind(
                                        &loop_paths,
                                        task_id,
                                        checkpoint_kind,
                                        &checkpoint_summary,
                                        &checkpoint_detail,
                                        Some(&output_text),
                                    ) {
                                        eprintln!(
                                            "failed to requeue context preparation task {task_id} as {checkpoint_kind}: {err}"
                                        );
                                    }
                                    task_outcome_persisted = true;
                                    task_status = "continue".to_string();
                                    normalized_next_mode = "reprioritize".to_string();
                                    record_context_preparation_checkpoint_event(
                                        &loop_paths,
                                        &task,
                                        turn_id,
                                        checkpoint_kind,
                                        preparation_artifact,
                                        &checkpoint_summary,
                                    );
                                }
                            } else if matches!(
                                task.task_kind.as_str(),
                                "worker_review" | "self_review"
                            ) {
                                let (review_resolution, fallback_note) = resolve_completion_review(
                                    &task_status,
                                    result.completion_review.as_ref(),
                                );
                                let effective_review_status = match review_resolution {
                                    CompletionReviewResolution::Approve => "done",
                                    CompletionReviewResolution::Block => "blocked",
                                    CompletionReviewResolution::Revise => "continue",
                                };
                                if let Some(note) = fallback_note {
                                    checkpoint_summary = format!(
                                        "Completion review reopened parent task #{}.",
                                        task.parent_task_id.unwrap_or(task.id)
                                    );
                                    checkpoint_detail = format!(
                                        "{}\n\nReview resolution:\n{}",
                                        checkpoint_detail, note
                                    );
                                }
                                task_status = effective_review_status.to_string();
                                normalized_next_mode = match review_resolution {
                                    CompletionReviewResolution::Approve => "review".to_string(),
                                    CompletionReviewResolution::Block => "blocked".to_string(),
                                    CompletionReviewResolution::Revise => {
                                        "execute_task".to_string()
                                    }
                                };
                                if let Err(err) = complete_review_task(
                                    &loop_paths,
                                    &task,
                                    effective_review_status,
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
                                            "Proactive contact validation was blocked; the temporary grosshirn boost falls back to kleinhirn.",
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
                                            "Proactive contact validation finished; the temporary grosshirn boost falls back to kleinhirn.",
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
                            } else if owner_interrupt_can_complete_directly(
                                &loop_paths,
                                &task,
                                &task_status,
                                &normalized_next_mode,
                                used_machine_path,
                                &output_text,
                            ) {
                                if checkpoint_summary.trim().is_empty() {
                                    checkpoint_summary = format!(
                                        "Owner interrupt #{} was answered directly.",
                                        task.id
                                    );
                                }
                                if let Err(err) = complete_task(
                                    &loop_paths,
                                    task_id,
                                    &checkpoint_summary,
                                    &checkpoint_detail,
                                    Some(&output_text),
                                ) {
                                    eprintln!(
                                        "failed to complete direct owner interrupt {task_id}: {err}"
                                    );
                                }
                                task_status = "done".to_string();
                                normalized_next_mode = "reprioritize".to_string();
                            } else if task_outcome_persisted {
                                // The context-preparation loop already persisted its own task transition.
                            } else if normalized_next_mode == "review" && task_status == "continue"
                            {
                                if let Err(err) = emit_self_review_task(
                                    &loop_paths,
                                    &task,
                                    &checkpoint_summary,
                                    &checkpoint_detail,
                                    Some(&output_text),
                                ) {
                                    eprintln!(
                                        "failed to emit self-review task for {task_id}: {err}"
                                    );
                                    let spawn_failure_summary = format!(
                                        "Completion review could not be queued for task #{}.",
                                        task.id
                                    );
                                    let spawn_failure_detail = format!(
                                        "{}\n\nThe self-review task could not be enqueued:\n{}\n\nThe main task stays open and was requeued for further work instead of being treated as done.",
                                        checkpoint_detail, err
                                    );
                                    if let Err(requeue_err) = requeue_task_with_checkpoint_kind(
                                        &loop_paths,
                                        task_id,
                                        "review_spawn_failed",
                                        &spawn_failure_summary,
                                        &spawn_failure_detail,
                                        Some(&output_text),
                                    ) {
                                        eprintln!(
                                            "failed to requeue task {task_id} after review spawn failure: {requeue_err}"
                                        );
                                    }
                                    checkpoint_summary = spawn_failure_summary;
                                    checkpoint_detail = spawn_failure_detail;
                                    normalized_next_mode = "execute_task".to_string();
                                    task_status = "continue".to_string();
                                } else {
                                    checkpoint_summary =
                                        format!("Task #{} queued a completion review.", task.id);
                                    checkpoint_detail = format!(
                                        "{}\n\nA prioritized self-review task was queued. The task is not done yet; completion now depends on explicit review approval.",
                                        checkpoint_detail
                                    );
                                    normalized_next_mode = "await_review".to_string();
                                    task_status = "continue".to_string();
                                }
                            } else if next_mode == "delegate" {
                                let contract = result.delegate_contract.clone().unwrap_or_else(|| {
                                    crate::agentic::DelegationContract {
                                        worker_kind: "specialist_worker".to_string(),
                                        contract_title: format!("Delegated step for task {}", task.id),
                                        contract_detail: checkpoint_detail.clone(),
                                        request_note: "Execute the implementation in a bounded way and return it for review.".to_string(),
                                    }
                                });
                                let _ = set_agent_mode(
                                    &loop_paths,
                                    "delegate",
                                    Some(task.id),
                                    &task.title,
                                    &format!(
                                        "Task {} is being delegated to worker {}.",
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
                                        let persist_result =
                                            persist_task_progress_with_boundary_preemption(
                                                &loop_paths,
                                                &task,
                                                &mut normalized_next_mode,
                                                &checkpoint_summary,
                                                &mut checkpoint_detail,
                                                Some(&output_text),
                                            );
                                        if let Err(err) = persist_result {
                                            eprintln!(
                                                "failed to persist continued task {task_id}: {err}"
                                            );
                                        }
                                    }
                                    "blocked" => {
                                        let _ = release_task_grosshirn_boost(
                                            &loop_paths,
                                            task_id,
                                            "The task was blocked in the bounded step; the temporary grosshirn boost falls back to kleinhirn.",
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
                                            eprintln!(
                                                "failed to emit self-review task for {task_id}: {err}"
                                            );
                                            let spawn_failure_summary = format!(
                                                "Completion review could not be queued for task #{}.",
                                                task.id
                                            );
                                            let spawn_failure_detail = format!(
                                                "{}\n\nThe self-review task could not be enqueued:\n{}\n\nThe main task stays open and was requeued for further work instead of being treated as done.",
                                                checkpoint_detail, err
                                            );
                                            if let Err(requeue_err) =
                                                requeue_task_with_checkpoint_kind(
                                                    &loop_paths,
                                                    task_id,
                                                    "review_spawn_failed",
                                                    &spawn_failure_summary,
                                                    &spawn_failure_detail,
                                                    Some(&output_text),
                                                )
                                            {
                                                eprintln!(
                                                    "failed to requeue task {task_id} after review spawn failure: {requeue_err}"
                                                );
                                            }
                                            checkpoint_summary = spawn_failure_summary;
                                            checkpoint_detail = spawn_failure_detail;
                                            normalized_next_mode = "execute_task".to_string();
                                            task_status = "continue".to_string();
                                        } else {
                                            checkpoint_summary = format!(
                                                "Task #{} queued a completion review.",
                                                task.id
                                            );
                                            checkpoint_detail = format!(
                                                "{}\n\nA prioritized self-review task was queued. The task is not done yet; completion now depends on explicit review approval.",
                                                checkpoint_detail
                                            );
                                            normalized_next_mode = "await_review".to_string();
                                            task_status = "continue".to_string();
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
                            let keep_task_attached = mode_keeps_task_active(effective_next_mode)
                                || normalized_next_mode == "delegate"
                                || task.task_kind == "worker_review";
                            let _ = set_agent_mode(
                                &loop_paths,
                                effective_next_mode,
                                if keep_task_attached {
                                    Some(task.id)
                                } else {
                                    None
                                },
                                &task.title,
                                &format!("Task {} finished one unified mode cycle.", task.id),
                            );

                            if !matches!(task.task_kind.as_str(), "self_review" | "worker_review") {
                                publish_task_result_to_origin(
                                    &loop_paths,
                                    &task,
                                    &task_status,
                                    &checkpoint_summary,
                                    &checkpoint_detail,
                                    &output_text,
                                    task_used_grosshirn,
                                );
                            }
                            if task.task_kind == "homepage_bridge"
                                && !checkpoint_summary.trim().is_empty()
                            {
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
                                    !matches!(
                                        task.task_kind.as_str(),
                                        "self_preservation" | "recovery"
                                    ),
                                );
                            }
                            if result.status != "ok" && result.status != "idle" {
                                eprintln!(
                                    "bounded task loop {}: {}",
                                    task.id,
                                    result.status_note()
                                );
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
                                let mut retry_mode = continue_next_mode_for_task(&task).to_string();
                                let retry_summary = format!(
                                    "Task {} returned empty model text and will be reclassified.",
                                    task.id
                                );
                                let mut retry_detail = format!(
                                    "{}\n\nThe bounded step was not treated as a real content block and remains in the same task loop for a cleaner retry.",
                                    detail
                                );
                                let persist_result = persist_task_progress_with_boundary_preemption(
                                    &loop_paths,
                                    &task,
                                    &mut retry_mode,
                                    &retry_summary,
                                    &mut retry_detail,
                                    None,
                                );
                                let _ = persist_result;
                                let _ = complete_agent_turn(
                                    &loop_paths,
                                    turn_id,
                                    "completed",
                                    &retry_mode,
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
                let last_context_progress_event_at = if task_kind == "context_preparation" {
                    Some(Instant::now())
                } else {
                    None
                };
                agentic_task = Some(RunningAgenticTask {
                    turn_id,
                    task_id,
                    task_title,
                    task_kind,
                    started_at: Instant::now(),
                    last_context_progress_event_at,
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
                let effective_stale_after_secs =
                    effective_live_turn_stale_after_secs(stale_after_secs, &running.task_kind);
                if age > effective_stale_after_secs {
                    loop_health = "stalled".to_string();
                    unhealthy_reason = Some(format!(
                        "active turn {} has been running for {}s",
                        running.turn_id, age
                    ));
                    thread_lifecycle = "stalled".to_string();
                    thread_note = format!(
                        "Turn {} for task {} is older than {}s and awaits watchdog recovery.",
                        running.turn_id, running.task_id, effective_stale_after_secs
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
                    thread_note = if running.task_kind == "context_preparation" {
                        if let Some(snapshot) =
                            context_preparation_progress_snapshot(&paths, running.task_id)
                        {
                            format!(
                                "Context optimization turn {} is active for task {} in phase {} iteration {}/{} with {} query answers.",
                                running.turn_id,
                                running.task_id,
                                snapshot.active_phase,
                                snapshot
                                    .phase_completed_loops
                                    .saturating_add(1)
                                    .min(snapshot.phase_max_loops),
                                snapshot.phase_max_loops,
                                snapshot.query_answers
                            )
                        } else {
                            format!(
                                "Context optimization turn {} is active for task {}.",
                                running.turn_id, running.task_id
                            )
                        }
                    } else {
                        format!(
                            "Turn {} is active for task {} in bounded execution.",
                            running.turn_id, running.task_id
                        )
                    };
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
                    thread_note = "Main infinity thread is actively protecting its own continuity."
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
    paths: &Paths,
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
    let repeated_same_summary = stage_policy.same_checkpoint_repeat_triggers_review && same_summary;
    let repeated_bounded_machine_step = same_summary
        && (checkpoint_summary.contains("Bounded command-exec executed:")
            || checkpoint_summary.contains("Exec-session action")
            || checkpoint_summary.contains("Bounded command-exec ausgefuehrt:")
            || checkpoint_summary.contains("Exec-Session-Aktion")
            || checkpoint_detail.contains("Bounded exec result:")
            || checkpoint_detail.contains("Exec session result:"));
    let context_preparation_task = task.task_kind == "context_preparation";
    let owner_interrupt_task = task.task_kind == "owner_interrupt";
    let owner_interrupt_under_grosshirn =
        owner_interrupt_task && task_has_active_grosshirn_boost(paths, task.id);
    let substantive_sticky_task =
        task_prefers_sticky_execute(task) && task.task_kind != "model_or_resource";
    let repeated_same_summary_streak = if repeated_same_summary {
        consecutive_matching_checkpoint_summaries(paths, task.id, checkpoint_summary)
    } else {
        0
    };
    let repeated_same_summary_escalates =
        repeated_same_summary && (!substantive_sticky_task || repeated_same_summary_streak >= 2);
    let context_preparation_total_limit = context_preparation_total_max_loops(paths) as i64;

    if context_preparation_task
        && !repeated_same_summary_escalates
        && !repeated_bounded_machine_step
        && task.run_count < context_preparation_total_limit
    {
        return None;
    }

    if repeated_bounded_machine_step
        || repeated_same_summary_escalates
        || (context_preparation_task && task.run_count >= context_preparation_total_limit)
        || (!context_preparation_task
            && !substantive_sticky_task
            && (!owner_interrupt_task || owner_interrupt_under_grosshirn)
            && task.run_count >= stage_policy.max_run_count_before_self_preservation_review)
    {
        let local_kleinhirn_decision_task = task.task_kind == "model_or_resource";
        let reason = if repeated_bounded_machine_step {
            "The same bounded machine action is repeating without new insight."
        } else if repeated_same_summary_escalates {
            "The last bounded response repeats the same checkpoint without new movement."
        } else if context_preparation_task {
            "Context optimization reached its maximum total iteration budget."
        } else if owner_interrupt_under_grosshirn {
            "The task has used too many bounded grosshirn attempts for the current maturity stage."
        } else {
            "The task has used too many bounded attempts for the current maturity stage."
        };
        let next_mode = if local_kleinhirn_decision_task {
            "review".to_string()
        } else if context_preparation_task {
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
        } else if context_preparation_task {
            "blocked".to_string()
        } else if owner_interrupt_task {
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
                "Task {} will not be continued blindly. {} Maturity stage: {}{}",
                task.id,
                reason,
                if context_preparation_task {
                    "context_preparation".to_string()
                } else {
                    stage_policy.stage.clone()
                },
                if local_kleinhirn_decision_task {
                    " Local kleinhirn upgrade stays on the local review path."
                } else if context_preparation_task {
                    " Context preparation must stop after the configured total iteration budget instead of slipping into newborn+N retries."
                } else if owner_interrupt_task {
                    " Owner interrupt stays on the direct owner-review path."
                } else {
                    ""
                }
            ),
            checkpoint_detail: format!(
                "{}\n\nPrevious checkpoint: {}\nCurrent checkpoint: {}\nRun count: {}\nStage: {}\nStage threshold before self-preservation review: {}\nHard block in this stage: {}\nForced local kleinhirn review path: {}",
                checkpoint_detail,
                task.last_checkpoint_summary.as_deref().unwrap_or("none"),
                checkpoint_summary,
                task.run_count,
                if context_preparation_task {
                    "context_preparation".to_string()
                } else {
                    stage_policy.stage.clone()
                },
                if context_preparation_task {
                    context_preparation_total_limit
                } else {
                    stage_policy.max_run_count_before_self_preservation_review
                },
                stage_policy.hard_block_unproductive_tasks,
                local_kleinhirn_decision_task
            ),
            spawn_self_preservation_task: !stage_policy.hard_block_unproductive_tasks
                && !local_kleinhirn_decision_task
                && !context_preparation_task
                && !owner_interrupt_task
                && !matches!(task.task_kind.as_str(), "self_preservation" | "recovery"),
        });
    }

    None
}

fn consecutive_matching_checkpoint_summaries(paths: &Paths, task_id: i64, summary: &str) -> usize {
    let normalized = summary.trim();
    list_task_checkpoints(paths, task_id, 3)
        .unwrap_or_default()
        .into_iter()
        .take_while(|checkpoint| checkpoint.summary.trim() == normalized)
        .count()
}

fn find_interrupt_preemption_candidate(
    paths: &Paths,
    active_task: &crate::runtime_db::TaskRecord,
    active_turn_id: i64,
) -> anyhow::Result<
    Option<(
        crate::runtime_db::TaskRecord,
        crate::runtime_db::TurnSignalRecord,
    )>,
> {
    let latest_interrupt_signal = list_recent_turn_signals(paths, 12)?
        .into_iter()
        .find(|signal| {
            signal.turn_id == Some(active_turn_id)
                && signal.task_id == Some(active_task.id)
                && signal.signal_kind == "interrupt"
                && signal.status == "recorded"
        });
    let Some(signal) = latest_interrupt_signal else {
        return Ok(None);
    };

    let owner_name = load_bios(paths).owner.name;
    let queued_interrupt_task = list_queued_tasks(paths, 16)?.into_iter().find(|task| {
        task.source_interrupt_id.is_some()
            && (task.priority_score > active_task.priority_score
                || queued_owner_interrupt_should_preempt_active_task(
                    task,
                    active_task,
                    &owner_name,
                ))
    });
    Ok(queued_interrupt_task.map(|task| (task, signal)))
}

fn queued_owner_interrupt_should_preempt_active_task(
    queued_task: &crate::runtime_db::TaskRecord,
    active_task: &crate::runtime_db::TaskRecord,
    owner_name: &str,
) -> bool {
    if queued_task.task_kind != "owner_interrupt"
        || queued_task.source_interrupt_id.is_none()
        || !matches!(
            queued_task.source_channel.as_str(),
            "terminal" | "attach_terminal" | "bios" | "homepage" | "email"
        )
        || !is_owner_speaker(&queued_task.speaker, owner_name)
        || queued_task.priority_score < active_task.priority_score
    {
        return false;
    }

    if active_task.task_kind != "owner_interrupt" {
        return true;
    }

    queued_task.id > active_task.id
        && queued_task.id != active_task.id
        && queued_task.source_interrupt_id != active_task.source_interrupt_id
}

fn is_owner_speaker(speaker: &str, owner_name: &str) -> bool {
    let speaker_norm = normalize_identity_for_owner_check(speaker);
    let owner_norm = normalize_identity_for_owner_check(owner_name);
    let canonical_owner_norm = normalize_identity_for_owner_check("Michael Welsch");
    if matches!(
        speaker_norm.as_str(),
        "owner" | "root_owner" | "bios_owner" | "initiator_owner"
    ) {
        return true;
    }
    if !owner_norm.is_empty() && speaker_norm == owner_norm {
        return true;
    }
    if !owner_norm.is_empty() && speaker_norm.contains(&owner_norm) {
        return true;
    }
    speaker_norm == canonical_owner_norm || speaker_norm.contains(&canonical_owner_norm)
}

fn normalize_identity_for_owner_check(value: &str) -> String {
    value
        .trim()
        .to_lowercase()
        .chars()
        .filter(|ch| ch.is_ascii_alphanumeric())
        .collect()
}

fn find_boundary_owner_interrupt_preemption(
    paths: &Paths,
    active_task: &crate::runtime_db::TaskRecord,
) -> anyhow::Result<Option<crate::runtime_db::TaskRecord>> {
    let owner_name = load_bios(paths).owner.name;
    Ok(list_queued_tasks(paths, 16)?.into_iter().find(|task| {
        queued_owner_interrupt_should_preempt_active_task(task, active_task, &owner_name)
    }))
}

fn persist_task_progress_with_boundary_preemption(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    next_mode: &mut String,
    checkpoint_summary: &str,
    checkpoint_detail: &mut String,
    output: Option<&str>,
) -> anyhow::Result<()> {
    if !mode_keeps_task_active(next_mode) {
        return requeue_task(
            paths,
            task.id,
            checkpoint_summary,
            checkpoint_detail,
            output,
        );
    }

    if let Some(queued_owner_task) = find_boundary_owner_interrupt_preemption(paths, task)? {
        let reason = format!(
            "Queued owner interrupt task {} from {} preempts task {} at the bounded-turn boundary before sticky {} continuation.",
            queued_owner_task.id, queued_owner_task.source_channel, task.id, next_mode
        );
        if !checkpoint_detail.contains(&reason) {
            checkpoint_detail.push_str("\n\nBoundary preemption:\n");
            checkpoint_detail.push_str(&reason);
        }
        *next_mode = "reprioritize".to_string();
        yield_active_task_for_preemption(paths, task.id, &reason)?;
        return requeue_task(
            paths,
            task.id,
            checkpoint_summary,
            checkpoint_detail,
            output,
        );
    }

    continue_active_task(
        paths,
        task.id,
        next_mode,
        checkpoint_summary,
        checkpoint_detail,
        output,
    )
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
        && checkpoint_summary.contains("Local kleinhirn upgrade stays on the local review path")
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
        .contains("local kleinhirn upgrade stays on the local review path")
        || checkpoint_detail_lower
            .contains("local kleinhirn upgrade stays on the local review path")
        || last_summary.contains("local kleinhirn upgrade stays on the local review path");

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

fn continue_next_mode_for_task(task: &crate::runtime_db::TaskRecord) -> &'static str {
    match execution_mode_for_task(task) {
        "self_preservation" => "self_preservation",
        "recovery" => "recovery",
        "historical_research" => "historical_research",
        "review" => "review",
        _ => "execute_task",
    }
}

fn task_prefers_sticky_execute(task: &crate::runtime_db::TaskRecord) -> bool {
    continue_next_mode_for_task(task) == "execute_task"
}

fn mode_keeps_task_active(mode: &str) -> bool {
    matches!(
        mode,
        "execute_task" | "self_preservation" | "recovery" | "historical_research"
    )
}

fn normalize_sticky_continue_mode(
    task: &crate::runtime_db::TaskRecord,
    task_status: &str,
    next_mode: &str,
) -> String {
    if task_status == "continue" && next_mode == "reprioritize" && task_prefers_sticky_execute(task)
    {
        continue_next_mode_for_task(task).to_string()
    } else {
        next_mode.to_string()
    }
}

fn should_enter_observe_mode(has_running_turn: bool, has_active_workstream: bool) -> bool {
    !has_running_turn && !has_active_workstream
}

fn should_run_reprioritize(has_active_workstream: bool) -> bool {
    !has_active_workstream
}

fn should_tick_agentic_loop(ticks: u64, has_active_workstream: bool) -> bool {
    has_active_workstream || ticks == 1 || ticks % 2 == 0
}

fn should_start_kleinhirn_repair_task(
    repair_running: bool,
    last_started_at: Option<Instant>,
    retry_cooldown: Duration,
) -> bool {
    if repair_running {
        return false;
    }
    match last_started_at {
        Some(started_at) => started_at.elapsed() >= retry_cooldown,
        None => true,
    }
}

fn email_sync_poll_interval() -> Duration {
    Duration::from_secs(
        std::env::var("CTO_AGENT_EMAIL_SYNC_INTERVAL_SECS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|secs| *secs >= 10)
            .unwrap_or(60),
    )
}

fn should_start_email_sync_task(
    sync_running: bool,
    last_started_at: Option<Instant>,
    poll_interval: Duration,
) -> bool {
    if sync_running {
        return false;
    }
    match last_started_at {
        Some(started_at) => started_at.elapsed() >= poll_interval,
        None => true,
    }
}

fn review_wait_recovery_interval() -> Duration {
    Duration::from_secs(
        std::env::var("CTO_AGENT_REVIEW_WAIT_RECOVERY_INTERVAL_SECS")
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|secs| *secs >= 5)
            .unwrap_or(30),
    )
}

fn should_start_review_wait_recovery(
    last_started_at: Option<Instant>,
    recovery_interval: Duration,
) -> bool {
    match last_started_at {
        Some(started_at) => started_at.elapsed() >= recovery_interval,
        None => true,
    }
}

fn task_kind_supports_task_local_brain_switch(task_kind: &str) -> bool {
    !matches!(
        task_kind,
        "self_preservation"
            | "recovery"
            | "historical_research"
            | "context_preparation"
            | "worker_review"
            | "self_review"
            | "proactive_contact_review"
    )
}

fn task_kind_uses_explicit_context_preparation(task_kind: &str) -> bool {
    let _ = task_kind;
    // Context distillation is now built inline with every normal context package.
    // The loop should execute substantive work directly instead of detouring through
    // a separate context_preparation task family.
    false
}

fn build_context_preparation_detail(
    task: &crate::runtime_db::TaskRecord,
    checkpoints: &[crate::runtime_db::TaskCheckpointRecord],
) -> String {
    let checkpoint_block = if checkpoints.is_empty() {
        "No recent checkpoints yet.".to_string()
    } else {
        checkpoints
            .iter()
            .take(3)
            .map(|checkpoint| {
                format!(
                    "- {}: {}",
                    checkpoint.checkpoint_kind,
                    checkpoint.summary.trim()
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    format!(
        "Prepare the ideal next-turn context for parent task #{id}.\n\nParent task:\nTitle: {title}\nKind: {kind}\nChannel: {channel}\nSpeaker: {speaker}\nPriority: {priority}\nDetail:\n{detail}\n\nRecent parent checkpoints:\n{checkpoint_block}\n\nDefinition of done for this preparation step:\n1. Name the most valuable immediate next direct step for the parent task.\n2. Verify or surface the highest-value missing context, artifact, contract, skill, path, or dependency for that next step.\n3. Leave behind a compact preparation artifact the parent task can consume in the next bounded run.\n4. Do not claim the parent task itself is solved unless the evidence really proves it.\n\nIf repository or filesystem context matters, use bounded exec work. If older decisions are missing, use targeted historical reload deliberately instead of vague context complaints.",
        id = task.id,
        title = task.title,
        kind = task.task_kind,
        channel = task.source_channel,
        speaker = task.speaker,
        priority = task.priority_score,
        detail = task.detail,
        checkpoint_block = checkpoint_block,
    )
}

fn classify_context_preparation_need(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
) -> anyhow::Result<ContextPreparationRouting> {
    if !task_kind_uses_explicit_context_preparation(&task.task_kind) {
        return Ok(ContextPreparationRouting::Proceed);
    }

    let checkpoints = list_task_checkpoints(paths, task.id, 8)?;
    if checkpoints
        .iter()
        .any(|checkpoint| checkpoint.checkpoint_kind == "context_preparation_ready")
    {
        return Ok(ContextPreparationRouting::Proceed);
    }

    if let Some(existing) = list_open_tasks(paths, 96)?.into_iter().find(|candidate| {
        candidate.task_kind == "context_preparation"
            && candidate.parent_task_id == Some(task.id)
            && matches!(
                candidate.status.as_str(),
                "queued" | "active" | "await_review"
            )
    }) {
        return Ok(ContextPreparationRouting::YieldToExisting {
            prep_task_id: existing.id,
            prep_title: existing.title,
        });
    }

    Ok(ContextPreparationRouting::EnqueueFresh {
        title: format!("Prepare context for task {}", task.id),
        detail: build_context_preparation_detail(task, &checkpoints),
        priority_score: task.priority_score.saturating_add(5),
    })
}

#[derive(Debug, Default)]
struct PreparedContextValidation {
    ready: bool,
    issues: Vec<String>,
    repeated_from_prior: bool,
    phase_limit_hit: bool,
}

fn approx_context_tokens(text: &str) -> usize {
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    words.max((chars + 3) / 4)
}

fn normalized_context_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_ascii_lowercase()
}

fn extract_prepared_context_artifact_from_detail(detail: &str) -> Option<ContextPreparedArtifact> {
    let value = serde_json::from_str::<serde_json::Value>(detail)
        .ok()
        .or_else(|| {
            for (start, ch) in detail.char_indices() {
                if ch != '{' {
                    continue;
                }
                let mut depth = 0_i64;
                let mut in_string = false;
                let mut escaped = false;
                for (offset, current) in detail[start..].char_indices() {
                    if in_string {
                        if escaped {
                            escaped = false;
                            continue;
                        }
                        match current {
                            '\\' => escaped = true,
                            '"' => in_string = false,
                            _ => {}
                        }
                        continue;
                    }
                    match current {
                        '"' => in_string = true,
                        '{' => depth += 1,
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                let end = start + offset + current.len_utf8();
                                if let Ok(value) =
                                    serde_json::from_str::<serde_json::Value>(&detail[start..end])
                                {
                                    return Some(value);
                                }
                                break;
                            }
                        }
                        _ => {}
                    }
                }
            }
            None
        })?;
    let artifact_value = value
        .get("preparedContextArtifact")
        .cloned()
        .or_else(|| value.get("review").map(|_| value.clone()))?;
    serde_json::from_value::<ContextPreparedArtifact>(artifact_value).ok()
}

fn append_prepared_context_artifact_to_detail(
    detail: &mut String,
    artifact: Option<&ContextPreparedArtifact>,
) {
    let Some(artifact) = artifact else {
        return;
    };
    if detail.contains("\"preparedContextArtifact\"") {
        return;
    }
    let Ok(serialized) = serde_json::to_string(&serde_json::json!({
        "preparedContextArtifact": artifact
    })) else {
        return;
    };
    if !detail.trim().is_empty() {
        detail.push_str("\n\n");
    }
    detail.push_str("Prepared context artifact:\n");
    detail.push_str(&serialized);
}

fn previous_prepared_context_artifact(
    checkpoints: &[crate::runtime_db::TaskCheckpointRecord],
) -> Option<ContextPreparedArtifact> {
    checkpoints
        .iter()
        .filter_map(|checkpoint| extract_prepared_context_artifact_from_detail(&checkpoint.detail))
        .next()
}

fn context_preparation_progress_snapshot(
    paths: &Paths,
    task_id: i64,
) -> Option<ContextPreparationProgressSnapshot> {
    let checkpoints = list_task_checkpoints(paths, task_id, 16).ok()?;
    let latest_package = crate::runtime_db::latest_context_package_for_task(paths, task_id)
        .ok()
        .flatten();
    let package_value = latest_package
        .as_ref()
        .and_then(|package| serde_json::from_str::<serde_json::Value>(&package.package_json).ok());
    let artifact = previous_prepared_context_artifact(&checkpoints);
    let checkpoint_kind = context_preparation_checkpoint_kind(artifact.as_ref());
    let active_phase = package_value
        .as_ref()
        .and_then(|value| value.pointer("/preparationContract/activePhase"))
        .and_then(serde_json::Value::as_str)
        .map(ToOwned::to_owned)
        .or_else(|| {
            package_value
                .as_ref()
                .and_then(|value| value.pointer("/contextOptimization/activePhase"))
                .and_then(serde_json::Value::as_str)
                .map(ToOwned::to_owned)
        })
        .unwrap_or_else(|| {
            artifact
                .as_ref()
                .map(|artifact| infer_current_context_preparation_phase(&checkpoints, artifact))
                .unwrap_or("query_plan")
                .to_string()
        });
    let context_mode = latest_package
        .as_ref()
        .map(|package| package.context_mode.clone())
        .unwrap_or_else(|| "preparation_query".to_string());
    let query_answers = package_value
        .as_ref()
        .and_then(|value| value.get("contextQueryAnswers"))
        .and_then(serde_json::Value::as_array)
        .map(|items| items.len())
        .unwrap_or(0);
    let phase_prefix = context_phase_prefix(checkpoint_kind);
    let phase_completed_loops = if phase_prefix.is_empty() {
        0
    } else {
        checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.checkpoint_kind.starts_with(phase_prefix))
            .count()
    };
    let optimization_policy = load_context_optimization_policy(paths);
    let phase_max_loops = optimization_policy
        .phases
        .iter()
        .find(|phase| phase.phase.eq_ignore_ascii_case(&active_phase))
        .map(|phase| phase.max_loops)
        .unwrap_or(4);
    let total_completed_loops = checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.checkpoint_kind.starts_with("context_"))
        .count();
    let latest_decision = artifact
        .as_ref()
        .map(|artifact| artifact.review.decision.clone())
        .unwrap_or_default();
    Some(ContextPreparationProgressSnapshot {
        active_phase,
        context_mode,
        query_answers,
        phase_completed_loops,
        phase_max_loops,
        total_completed_loops,
        total_max_loops: optimization_policy.max_total_loops.max(1),
        latest_decision,
    })
}

fn record_context_preparation_started_event(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    turn_id: i64,
) {
    let snapshot = context_preparation_progress_snapshot(paths, task.id);
    let body = if let Some(snapshot) = snapshot.as_ref() {
        format!(
            "Context optimization started: phase {} total iteration {}/{} for task #{}.",
            snapshot.active_phase,
            snapshot
                .total_completed_loops
                .saturating_add(1)
                .min(snapshot.total_max_loops),
            snapshot.total_max_loops,
            task.id
        )
    } else {
        format!("Context optimization started for task #{}.", task.id)
    };
    let payload = serde_json::json!({
        "turnId": turn_id,
        "taskId": task.id,
        "stage": "started",
        "taskKind": task.task_kind,
        "activePhase": snapshot.as_ref().map(|item| item.active_phase.clone()).unwrap_or_else(|| "query_plan".to_string()),
        "contextMode": snapshot.as_ref().map(|item| item.context_mode.clone()).unwrap_or_else(|| "preparation_query".to_string()),
        "phaseCompletedLoops": snapshot.as_ref().map(|item| item.phase_completed_loops).unwrap_or(0),
        "phaseMaxLoops": snapshot.as_ref().map(|item| item.phase_max_loops).unwrap_or(4),
        "totalCompletedLoops": snapshot.as_ref().map(|item| item.total_completed_loops).unwrap_or(0),
        "totalMaxLoops": snapshot.as_ref().map(|item| item.total_max_loops).unwrap_or(4),
        "queryAnswers": snapshot.as_ref().map(|item| item.query_answers).unwrap_or(0),
    });
    let _ = crate::runtime_db::record_agent_event(
        paths,
        "context/progress",
        Some(task.id),
        &task.title,
        &body,
        &payload.to_string(),
    );
}

fn record_context_preparation_checkpoint_event(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    turn_id: i64,
    checkpoint_kind: &str,
    artifact: Option<&ContextPreparedArtifact>,
    summary: &str,
) {
    let snapshot = context_preparation_progress_snapshot(paths, task.id);
    let phase_name = context_phase_name_from_checkpoint_kind(checkpoint_kind);
    let active_phase = snapshot
        .as_ref()
        .map(|item| item.active_phase.clone())
        .unwrap_or_else(|| phase_name.to_string());
    let phase_max_loops = snapshot
        .as_ref()
        .map(|item| item.phase_max_loops)
        .unwrap_or(4);
    let phase_completed_loops = snapshot
        .as_ref()
        .map(|item| item.phase_completed_loops)
        .unwrap_or(0);
    let latest_decision = artifact
        .map(|artifact| artifact.review.decision.clone())
        .or_else(|| snapshot.as_ref().map(|item| item.latest_decision.clone()))
        .unwrap_or_default();
    let latest_package = crate::runtime_db::latest_context_package_for_task(paths, task.id)
        .ok()
        .flatten();
    let body = format!(
        "Context optimization checkpoint: {} total iteration {}/{} for task #{}. {}",
        active_phase,
        snapshot
            .as_ref()
            .map(|item| item.total_completed_loops)
            .unwrap_or(phase_completed_loops)
            .min(
                snapshot
                    .as_ref()
                    .map(|item| item.total_max_loops)
                    .unwrap_or(phase_max_loops)
                    .max(1),
            ),
        snapshot
            .as_ref()
            .map(|item| item.total_max_loops)
            .unwrap_or(phase_max_loops),
        task.id,
        concise_reply_line(summary)
    );
    let payload = serde_json::json!({
        "turnId": turn_id,
        "taskId": task.id,
        "stage": "checkpoint",
        "checkpointKind": checkpoint_kind,
        "activePhase": active_phase,
        "phaseCompletedLoops": phase_completed_loops,
        "phaseMaxLoops": phase_max_loops,
        "totalCompletedLoops": snapshot.as_ref().map(|item| item.total_completed_loops).unwrap_or(phase_completed_loops),
        "totalMaxLoops": snapshot.as_ref().map(|item| item.total_max_loops).unwrap_or(phase_max_loops),
        "decision": latest_decision,
        "questionCount": artifact.map(|item| item.questions.len()).unwrap_or(0),
        "blockCount": artifact.map(|item| item.blocks.iter().filter(|block| !block.content.trim().is_empty()).count()).unwrap_or(0),
        "hasDistilledFocus": latest_package
            .as_ref()
            .and_then(|package| serde_json::from_str::<serde_json::Value>(&package.package_json).ok())
            .and_then(|value| value.get("contextDistillation").cloned())
            .is_some(),
        "missingEvidence": artifact.map(|item| item.review.missing_evidence.clone()).unwrap_or_default(),
        "weakBlocks": artifact.map(|item| item.review.weak_blocks.clone()).unwrap_or_default(),
    });
    let _ = crate::runtime_db::record_agent_event(
        paths,
        "context/progress",
        Some(task.id),
        &task.title,
        &body,
        &payload.to_string(),
    );
}

fn record_context_preparation_active_event(
    paths: &Paths,
    task_id: i64,
    task_title: &str,
    turn_id: i64,
    age_secs: u64,
) {
    let snapshot = context_preparation_progress_snapshot(paths, task_id);
    let body = if let Some(snapshot) = snapshot.as_ref() {
        format!(
            "Context optimization still running: phase {} total iteration {}/{} for task #{} after {}s with {} query answers.",
            snapshot.active_phase,
            snapshot
                .total_completed_loops
                .saturating_add(1)
                .min(snapshot.total_max_loops),
            snapshot.total_max_loops,
            task_id,
            age_secs,
            snapshot.query_answers
        )
    } else {
        format!(
            "Context optimization still running for task #{} after {}s.",
            task_id, age_secs
        )
    };
    let payload = serde_json::json!({
        "turnId": turn_id,
        "taskId": task_id,
        "stage": "active",
        "ageSecs": age_secs,
        "activePhase": snapshot.as_ref().map(|item| item.active_phase.clone()).unwrap_or_else(|| "query_plan".to_string()),
        "contextMode": snapshot.as_ref().map(|item| item.context_mode.clone()).unwrap_or_else(|| "preparation_query".to_string()),
        "phaseCompletedLoops": snapshot.as_ref().map(|item| item.phase_completed_loops).unwrap_or(0),
        "phaseMaxLoops": snapshot.as_ref().map(|item| item.phase_max_loops).unwrap_or(4),
        "totalCompletedLoops": snapshot.as_ref().map(|item| item.total_completed_loops).unwrap_or(0),
        "totalMaxLoops": snapshot.as_ref().map(|item| item.total_max_loops).unwrap_or(4),
        "queryAnswers": snapshot.as_ref().map(|item| item.query_answers).unwrap_or(0),
        "decision": snapshot.as_ref().map(|item| item.latest_decision.clone()).unwrap_or_default(),
    });
    let _ = crate::runtime_db::record_agent_event(
        paths,
        "context/progress",
        Some(task_id),
        task_title,
        &body,
        &payload.to_string(),
    );
}

fn context_phase_prefix(checkpoint_kind: &str) -> &'static str {
    match checkpoint_kind {
        "context_query_plan" | "context_query_refine" => "context_query",
        "context_rewrite_draft" => "context_rewrite",
        "context_rewrite_review" => "context_rewrite_review",
        "context_preparation_ready" => "context_preparation_ready",
        _ => "",
    }
}

fn context_phase_name_from_checkpoint_kind(checkpoint_kind: &str) -> &'static str {
    match checkpoint_kind {
        "context_query_plan" | "context_query_refine" => "query_plan",
        "context_rewrite_draft" => "rewrite",
        "context_rewrite_review" => "review",
        "context_preparation_ready" => "ready",
        _ => "",
    }
}

fn context_preparation_checkpoint_kind_for_phase(active_phase: &str) -> &'static str {
    if active_phase.eq_ignore_ascii_case("rewrite") {
        "context_rewrite_draft"
    } else if active_phase.eq_ignore_ascii_case("review") {
        "context_rewrite_review"
    } else {
        "context_query_plan"
    }
}

fn artifacts_repeat_without_progress(
    current: &ContextPreparedArtifact,
    previous: Option<&ContextPreparedArtifact>,
) -> bool {
    let Some(previous) = previous else {
        return false;
    };
    let current_blocks = current
        .blocks
        .iter()
        .filter(|block| !block.content.trim().is_empty())
        .map(|block| {
            (
                block.block_id.as_str(),
                normalized_context_text(&block.content),
                block.evidence_refs.clone(),
            )
        })
        .collect::<Vec<_>>();
    let previous_blocks = previous
        .blocks
        .iter()
        .filter(|block| !block.content.trim().is_empty())
        .map(|block| {
            (
                block.block_id.as_str(),
                normalized_context_text(&block.content),
                block.evidence_refs.clone(),
            )
        })
        .collect::<Vec<_>>();
    !current_blocks.is_empty()
        && current.immediate_next_step.trim() == previous.immediate_next_step.trim()
        && current_blocks == previous_blocks
}

fn weak_blocks_made_progress(
    current: &ContextPreparedArtifact,
    previous: Option<&ContextPreparedArtifact>,
) -> bool {
    let Some(previous) = previous else {
        return true;
    };
    if previous.review.weak_blocks.is_empty() {
        return true;
    }
    let current_blocks = current
        .blocks
        .iter()
        .map(|block| {
            (
                block.block_id.as_str(),
                normalized_context_text(&block.content),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    let previous_blocks = previous
        .blocks
        .iter()
        .map(|block| {
            (
                block.block_id.as_str(),
                normalized_context_text(&block.content),
            )
        })
        .collect::<std::collections::HashMap<_, _>>();
    previous.review.weak_blocks.iter().any(|block_id| {
        current_blocks.get(block_id.as_str()) != previous_blocks.get(block_id.as_str())
    })
}

fn infer_current_context_preparation_phase(
    checkpoints: &[crate::runtime_db::TaskCheckpointRecord],
    artifact: &ContextPreparedArtifact,
) -> &'static str {
    let previous = previous_prepared_context_artifact(checkpoints);
    let Some(previous) = previous.as_ref() else {
        return "query_plan";
    };
    let previous_decision = previous.review.decision.trim().to_ascii_lowercase();
    if previous.questions.is_empty() {
        "query_plan"
    } else if previous.blocks.is_empty() {
        if artifact.blocks.is_empty() {
            "query_plan"
        } else {
            "rewrite"
        }
    } else if previous_decision.contains("query") && artifact.blocks.is_empty() {
        "query_plan"
    } else {
        "review"
    }
}

fn validate_prepared_context_artifact(
    artifact: &ContextPreparedArtifact,
    checkpoints: &[crate::runtime_db::TaskCheckpointRecord],
    paths: &Paths,
) -> PreparedContextValidation {
    let policy = load_context_optimization_policy(paths);
    let mut issues = Vec::new();
    let decision = artifact.review.decision.trim().to_ascii_lowercase();
    let active_phase = infer_current_context_preparation_phase(checkpoints, artifact);
    if artifact.immediate_next_step.trim().is_empty() {
        issues.push("missing immediate next step".to_string());
    }
    let phase_policy = policy
        .phases
        .iter()
        .find(|phase| phase.phase.eq_ignore_ascii_case(active_phase));
    let phase_requires_blocks = phase_policy
        .map(|phase| {
            phase
                .required_outputs
                .iter()
                .any(|output| output == "blocks")
        })
        .unwrap_or(false);
    let should_enforce_required_blocks = decision == "go" || phase_requires_blocks;
    let should_flag_missing_evidence = decision == "go";
    let should_flag_budget_violations = decision == "go" || phase_requires_blocks;
    if should_flag_missing_evidence && !artifact.review.missing_evidence.is_empty() {
        issues.push("review still reports missing evidence".to_string());
    }
    if should_flag_budget_violations && !artifact.review.budget_violations.is_empty() {
        issues.push("review still reports budget violations".to_string());
    }
    let block_policies = policy
        .blocks
        .iter()
        .map(|block| (block.block_id.as_str(), block))
        .collect::<std::collections::HashMap<_, _>>();
    let block_ids = artifact
        .blocks
        .iter()
        .filter(|block| !block.content.trim().is_empty())
        .map(|block| block.block_id.as_str())
        .collect::<std::collections::HashSet<_>>();
    if let Some(phase_policy) = phase_policy {
        for output in &phase_policy.required_outputs {
            match output.as_str() {
                "questions" if artifact.questions.is_empty() && decision != "blocked" => {
                    issues.push(format!(
                        "phase `{}` requires at least one question",
                        active_phase
                    ));
                }
                "blocks"
                    if artifact
                        .blocks
                        .iter()
                        .all(|block| block.content.trim().is_empty())
                        && decision != "blocked" =>
                {
                    issues.push(format!(
                        "phase `{}` requires rewritten blocks",
                        active_phase
                    ));
                }
                "review" if artifact.review.note.trim().is_empty() => {
                    issues.push(format!("phase `{}` is missing a review note", active_phase));
                }
                _ => {}
            }
        }
        if !phase_policy.allowed_review_decisions.is_empty()
            && !phase_policy
                .allowed_review_decisions
                .iter()
                .any(|allowed| allowed.eq_ignore_ascii_case(&decision))
        {
            issues.push(format!(
                "phase `{}` does not allow review decision `{}`",
                active_phase, decision
            ));
        }
    }
    if active_phase == "query_plan" && !artifact.blocks.is_empty() {
        issues.push("query_plan phase must not emit rewritten blocks yet".to_string());
    }
    if active_phase == "rewrite" && decision == "go" {
        issues.push("rewrite phase must hand off to review before `go`".to_string());
    }
    if should_enforce_required_blocks {
        for required in &policy.required_block_ids {
            if !block_ids.contains(required.as_str()) {
                issues.push(format!("missing required block `{required}`"));
            }
        }
    }
    let mut duplicate_blocks = std::collections::HashSet::new();
    for block in &artifact.blocks {
        if !duplicate_blocks.insert(block.block_id.as_str()) {
            issues.push(format!("duplicate block `{}`", block.block_id));
        }
        if block.content.trim().is_empty() {
            continue;
        }
        if block.why_included.trim().is_empty() {
            issues.push(format!(
                "block `{}` is missing `whyIncluded`",
                block.block_id
            ));
        }
        if block.evidence_refs.is_empty() {
            issues.push(format!("block `{}` has no `evidenceRefs`", block.block_id));
        }
        if let Some(policy_block) = block_policies.get(block.block_id.as_str()) {
            let approx_tokens = block
                .approx_tokens
                .unwrap_or_else(|| approx_context_tokens(&block.content));
            if approx_tokens > policy_block.token_budget {
                issues.push(format!(
                    "block `{}` exceeds token budget ({} > {})",
                    block.block_id, approx_tokens, policy_block.token_budget
                ));
            }
        }
    }
    let checkpoint_kind = context_preparation_checkpoint_kind(Some(artifact));
    let phase_prefix = context_phase_prefix(checkpoint_kind);
    let phase_limit_hit = if phase_prefix.is_empty() {
        false
    } else {
        let prior_loops = checkpoints
            .iter()
            .filter(|checkpoint| checkpoint.checkpoint_kind.starts_with(phase_prefix))
            .count();
        let phase_name = context_phase_name_from_checkpoint_kind(checkpoint_kind);
        policy
            .phases
            .iter()
            .find(|phase| phase.phase == phase_name)
            .map(|phase| prior_loops >= phase.max_loops)
            .unwrap_or(false)
    };
    if phase_limit_hit {
        issues.push(format!(
            "phase loop limit reached for `{}`",
            checkpoint_kind
        ));
    }
    let repeated_from_prior = artifacts_repeat_without_progress(
        artifact,
        previous_prepared_context_artifact(checkpoints).as_ref(),
    );
    if !weak_blocks_made_progress(
        artifact,
        previous_prepared_context_artifact(checkpoints).as_ref(),
    ) && decision != "go"
    {
        issues.push("review asked for weak-block revision but the new artifact does not improve any weak block".to_string());
    }
    if repeated_from_prior && decision != "go" {
        issues.push("prepared context repeated the prior artifact without progress".to_string());
    }
    PreparedContextValidation {
        ready: decision == "go" && issues.is_empty(),
        issues,
        repeated_from_prior,
        phase_limit_hit,
    }
}

fn context_preparation_checkpoint_kind(artifact: Option<&ContextPreparedArtifact>) -> &'static str {
    let Some(artifact) = artifact else {
        return "context_query_plan";
    };
    let decision = artifact.review.decision.trim().to_ascii_lowercase();
    if artifact.questions.is_empty() {
        "context_query_plan"
    } else if decision.contains("query") {
        "context_query_refine"
    } else if artifact.blocks.is_empty() {
        "context_rewrite_draft"
    } else if decision == "go" {
        "context_preparation_ready"
    } else {
        "context_rewrite_review"
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
            _ => continue_next_mode_for_task(task),
        },
        "blocked" => "blocked",
        _ => match task.task_kind.as_str() {
            "self_preservation" | "recovery" | "historical_research" => "reprioritize",
            "worker_review" => "review",
            _ => "review",
        },
    }
}

fn should_enqueue_history_research(task_kind: &str, needs_history_research: bool) -> bool {
    needs_history_research
        && task_kind != "historical_research"
        && task_kind != "context_preparation"
        && !matches!(
            task_kind,
            "self_review" | "worker_review" | "proactive_contact_review"
        )
}

fn should_keep_history_research_inline(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    needs_history_research: bool,
    checkpoint_detail: &str,
) -> bool {
    if !needs_history_research {
        return false;
    }

    task_prefers_sticky_execute(task)
        || (workspace_execution_contract_active(paths, task.id)
            && checkpoint_detail.contains("Bounded exec result:"))
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
    let stored =
        store_learning_entries(paths, task, turn_id, learning_entries, entry_status, source)?;
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
        "{} Learning path extended by {} entries.",
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
                    "Send the approved proactive contact now through the available channel autonomously.\n\nPerson: {person}\nEmail: {email}\nIntended channel: {channel}\nSubject: {subject}\n\nValidation note:\n{validation_note}\n\nRationale:\n{rationale}\n\nConflict check:\n{conflict_check}\n\nDraft:\n{body}",
                    person = candidate.person_display_name,
                    email = if candidate.person_email.trim().is_empty() {
                        "unknown"
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
                        "Send proactive contact to {}",
                        candidate.person_display_name
                    ),
                    &dispatch_detail,
                    640,
                )?;
                let _ = attach_dispatch_task_to_candidate(paths, candidate.id, dispatch_task.id)?;
                *checkpoint_summary = format!(
                    "{} A dispatch task for {} was queued.",
                    checkpoint_summary, candidate.person_display_name
                );
                *checkpoint_detail = format!(
                    "{}\n- dispatchTaskId: {}",
                    checkpoint_detail, dispatch_task.id
                );
            }
            *checkpoint_summary = format!(
                "{} Proactive contact for {} was validated with status {}.",
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
            "Review this proactive contact proposal before any dispatch.\n\nPerson: {person}\nEmail: {email}\nChannel: {channel}\nSubject: {subject}\n\nBenefit hypothesis:\n{rationale}\n\nConflict-of-interest check:\n{conflict}\n\nDraft:\n{body}\n\nReturn proactiveContactValidation with approve, reject, or revise. Approve only if the proposal truly serves the person's interest.",
            person = candidate.person_display_name,
            email = if candidate.person_email.trim().is_empty() {
                "unknown"
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
                "Validate proactive proposal for {}",
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
                "Proactive people contacts must be grosshirn-validated before dispatch.",
                1120,
            );
        }
    }
    *checkpoint_summary = format!(
        "{} A proactive contact proposal for {} was recorded for validation.",
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
            .unwrap_or_else(|| "unknown".to_string()),
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
        let summary = format!("Dispatch task {} has no bound proactive contact.", task.id);
        block_task(paths, task.id, &summary, &summary, None)?;
        return Ok(());
    };

    if candidate.status == "sent" {
        complete_task(
            paths,
            task.id,
            &format!(
                "Proactive contact to {} had already been sent.",
                candidate.person_display_name
            ),
            &format!(
                "Candidate {} was already sent on {}.",
                candidate.id,
                candidate.dispatched_at.as_deref().unwrap_or("unknown")
            ),
            None,
        )?;
        return Ok(());
    }

    if candidate.status != "approved" {
        let summary = format!(
            "Proactive contact to {} is not ready for dispatch.",
            candidate.person_display_name
        );
        let detail = format!(
            "Candidate {} has status {} instead of approved.",
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
                    "Proactive contact to {} could not be sent.",
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
                    "Proactive contact to {person} was sent autonomously via {channel}.\n\nSubject: {subject}\nMessage-ID: {message_id}\nDispatch note: {note}",
                    person = final_candidate.person_display_name,
                    channel = final_candidate.dispatch_channel,
                    subject = final_candidate.subject,
                    message_id = if final_candidate.outbound_message_id.trim().is_empty() {
                        "unknown"
                    } else {
                        final_candidate.outbound_message_id.as_str()
                    },
                    note = final_candidate.dispatch_note,
                );
                complete_task(
                    paths,
                    task.id,
                    &format!(
                        "Proactive contact to {} was sent.",
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
                        "Proactive contact to {} could not be sent.",
                        candidate.person_display_name
                    ),
                    &detail,
                    None,
                )?;
            }
        },
        _ => {
            let detail = format!(
                "Unsupported dispatch channel {} for candidate {}.",
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
                    "Proactive contact to {} remained blocked.",
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
                    "Email delivery is intended, but {} has no email address.",
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
                    "There is no autonomous dispatch path for channel {} yet, and no email address is known for {}.",
                    candidate.channel,
                    candidate.person_display_name
                );
            }
            Ok(ProactiveDispatchRoute {
                channel: "email".to_string(),
                recipient,
                route_note: format!(
                    "No direct dispatch path exists for {}; use the existing email address as a controlled fallback.",
                    candidate.channel
                ),
            })
        }
        _ => {
            if recipient.is_empty() {
                anyhow::bail!(
                    "Unknown dispatch channel {} without a stored email address for {}.",
                    candidate.channel,
                    candidate.person_display_name
                );
            }
            Ok(ProactiveDispatchRoute {
                channel: "email".to_string(),
                recipient,
                route_note: format!(
                    "Channel {} is not integrated as an active dispatch adapter yet; use the email fallback.",
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
    let mail_tooling = ensure_mail_reply_path_ready(paths)?;
    let mail_env = load_runtime_mail_env(paths)?;
    let directive = ExecCommandDirective {
        command: vec![
            "env".to_string(),
            format!("CTO_EMAIL_ADDRESS={}", mail_env.address),
            format!("CTO_EMAIL_PASSWORD={}", mail_env.password),
            "node".to_string(),
            mail_tooling.script_path.display().to_string(),
            "--db".to_string(),
            paths.runtime_db_path.display().to_string(),
            "send".to_string(),
            "--smtp-host".to_string(),
            mail_env.smtp_host.clone(),
            "--smtp-port".to_string(),
            mail_env.smtp_port.to_string(),
            "--to".to_string(),
            route.recipient.clone(),
            "--subject".to_string(),
            candidate.subject.clone(),
            "--body".to_string(),
            candidate.draft_body.clone(),
        ],
        workdir: Some(paths.root.display().to_string()),
        timeout_ms: Some(45_000),
        justification: Some(
            "Send an approved proactive person contact through the local mail path.".to_string(),
        ),
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
        anyhow::bail!("Mail delivery failed: {error}");
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
        anyhow::bail!("Mail delivery returned without ok=true: {error}");
    }
    let message_id = payload
        .as_ref()
        .and_then(|value| value.get("message_id").or_else(|| value.get("messageId")))
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| anyhow::anyhow!("Mail client replied without message_id."))?;
    let account_id = payload
        .as_ref()
        .and_then(|value| value.get("account_id").or_else(|| value.get("accountKey")))
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    Ok((
        message_id,
        format!(
            "Local mail client sent via account {} to {}. Reply path stays on {} with interrupt-capable sync via {}.",
            account_id,
            route.recipient,
            path_display_name(&mail_tooling.script_path),
            path_display_name(&mail_tooling.agent_binary_path)
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
        "Autonomous delivery of an approved proactive contact failed or was blocked.\n\nPerson: {person}\nEmail: {email}\nIntended channel: {channel}\nSubject: {subject}\n\nError:\n{failure}\n\nCheck the mail path, credentials, channel adapters, or conflict situation and derive bounded repair work or an alternative safe contact path.",
        person = candidate.person_display_name,
        email = if candidate.person_email.trim().is_empty() {
            "unknown"
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
        &format!("Repair dispatch path for {}", candidate.person_display_name),
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

fn command_uses_reviewed_js_mail_send(command: &[String]) -> bool {
    if command.is_empty() {
        return false;
    }
    let joined = command.join(" ").to_lowercase();
    joined.contains("communication_mail_cli.mjs")
        && command.iter().any(|part| part.eq_ignore_ascii_case("send"))
}

fn command_uses_legacy_raw_mail_send(command: &[String]) -> bool {
    if command.is_empty() {
        return false;
    }
    let joined = command.join(" ").to_lowercase();
    (joined.contains("smtplib")
        || joined.contains("send.one.com")
        || joined.contains("smtp.")
        || joined.contains("mail_sent"))
        && !command_uses_reviewed_js_mail_send(command)
}

fn command_option_value<'a>(command: &'a [String], flag: &str) -> Option<&'a str> {
    command
        .windows(2)
        .find(|window| window[0] == flag)
        .map(|window| window[1].as_str())
}

fn visible_owner_mail_send_completion(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    directive: &ExecCommandDirective,
    exec_result: &crate::tooling::ExecCommandResult,
) -> Option<(String, String)> {
    if task.task_kind != "owner_interrupt" || exec_result.status != "ok" || exec_result.timed_out {
        return None;
    }

    if command_uses_reviewed_js_mail_send(&directive.command) {
        let payload = parse_json_command_output(&exec_result.stdout)?;
        let ok = payload.get("ok").and_then(Value::as_bool).unwrap_or(false);
        let message_id = payload
            .get("message_id")
            .or_else(|| payload.get("messageId"))
            .and_then(Value::as_str)
            .filter(|value| !value.trim().is_empty())?;
        if !ok {
            return None;
        }
        let stored_in_outbox =
            crate::runtime_db::communication_message_sent(paths, message_id).ok()?;
        if !stored_in_outbox {
            return None;
        }
        let recipient = command_option_value(&directive.command, "--to").unwrap_or("unknown");
        let subject =
            command_option_value(&directive.command, "--subject").unwrap_or("(ohne Betreff)");
        return Some((
            format!(
                "Visible owner email reply sent successfully to {}.",
                recipient
            ),
            format!(
                "The bounded owner mail step completed through the reviewed JS mail adapter.\nRecipient: {recipient}\nSubject: {subject}\nMessage-ID: {message_id}\nCWD: {cwd}\nSTDOUT:\n{stdout}\n\nSTDERR:\n{stderr}",
                recipient = recipient,
                subject = subject,
                message_id = message_id,
                cwd = exec_result.cwd,
                stdout = exec_result.stdout,
                stderr = exec_result.stderr,
            ),
        ));
    }

    None
}

struct RuntimeMailEnv {
    address: String,
    password: String,
    imap_host: String,
    imap_port: u16,
    smtp_host: String,
    smtp_port: u16,
}

struct MailSyncOutcome {
    fetched_count: i64,
    stored_count: i64,
    note: String,
}

struct MailReplyPathReadiness {
    script_path: PathBuf,
    agent_binary_path: PathBuf,
}

fn ensure_mail_reply_path_ready(paths: &Paths) -> anyhow::Result<MailReplyPathReadiness> {
    let script_path = paths.root.join("scripts/communication_mail_cli.mjs");
    if !script_path.exists() {
        anyhow::bail!(
            "Outbound email blocked because the JS mail client {} is missing.",
            path_display_name(&script_path)
        );
    }
    let schema_path = paths.root.join("scripts/communication_schema.sql");
    if !schema_path.exists() {
        anyhow::bail!(
            "Outbound email blocked because the communication schema {} is missing.",
            path_display_name(&schema_path)
        );
    }
    let _ = load_runtime_mail_env(paths)?;
    let release_binary = paths.root.join("target/release/cto-agent");
    let debug_binary = paths.root.join("target/debug/cto-agent");
    let agent_binary_path = if release_binary.exists() {
        release_binary
    } else if debug_binary.exists() {
        debug_binary
    } else {
        anyhow::bail!(
            "Outbound email blocked because no cto-agent binary is available for the inbound interrupt bridge (expected {} or {}).",
            path_display_name(&paths.root.join("target/release/cto-agent")),
            path_display_name(&paths.root.join("target/debug/cto-agent"))
        );
    };
    Ok(MailReplyPathReadiness {
        script_path,
        agent_binary_path,
    })
}

fn load_runtime_mail_env(paths: &Paths) -> anyhow::Result<RuntimeMailEnv> {
    let env_path = paths.root.join("runtime/kleinhirn.env");
    let text = fs::read_to_string(&env_path)
        .with_context(|| format!("failed to read {}", env_path.display()))?;
    let env_map = parse_runtime_env_text(&text);
    let address = std::env::var("CTO_EMAIL_ADDRESS")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| env_map.get("CTO_EMAIL_ADDRESS").cloned())
        .ok_or_else(|| anyhow::anyhow!("CTO_EMAIL_ADDRESS is missing in runtime env."))?;
    let password = std::env::var("CTO_EMAIL_PASSWORD")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| env_map.get("CTO_EMAIL_PASSWORD").cloned())
        .ok_or_else(|| anyhow::anyhow!("CTO_EMAIL_PASSWORD is missing in runtime env."))?;
    let imap_host = std::env::var("CTO_EMAIL_IMAP_HOST")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| env_map.get("CTO_EMAIL_IMAP_HOST").cloned())
        .unwrap_or_else(|| "imap.one.com".to_string());
    let imap_port = std::env::var("CTO_EMAIL_IMAP_PORT")
        .ok()
        .and_then(|raw| raw.parse::<u16>().ok())
        .or_else(|| {
            env_map
                .get("CTO_EMAIL_IMAP_PORT")
                .and_then(|raw| raw.parse::<u16>().ok())
        })
        .unwrap_or(993);
    let smtp_host = std::env::var("CTO_EMAIL_SMTP_HOST")
        .ok()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| env_map.get("CTO_EMAIL_SMTP_HOST").cloned())
        .unwrap_or_else(|| "send.one.com".to_string());
    let smtp_port = std::env::var("CTO_EMAIL_SMTP_PORT")
        .ok()
        .and_then(|raw| raw.parse::<u16>().ok())
        .or_else(|| {
            env_map
                .get("CTO_EMAIL_SMTP_PORT")
                .and_then(|raw| raw.parse::<u16>().ok())
        })
        .unwrap_or(465);
    Ok(RuntimeMailEnv {
        address,
        password,
        imap_host,
        imap_port,
        smtp_host,
        smtp_port,
    })
}

fn run_periodic_email_interrupt_sync(paths: &Paths) -> anyhow::Result<MailSyncOutcome> {
    let mail_tooling = ensure_mail_reply_path_ready(paths)?;
    let mail_env = load_runtime_mail_env(paths)?;
    let baseline_only = crate::runtime_db::communication_email_sync_needs_baseline(paths)?;
    let limit = std::env::var("CTO_AGENT_EMAIL_SYNC_LIMIT")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value >= 1 && *value <= 200)
        .unwrap_or(20);
    let output = Command::new("node")
        .arg(mail_tooling.script_path.as_os_str())
        .arg("sync")
        .arg("--db")
        .arg(paths.runtime_db_path.as_os_str())
        .arg("--imap-host")
        .arg(&mail_env.imap_host)
        .arg("--imap-port")
        .arg(mail_env.imap_port.to_string())
        .arg("--folder")
        .arg("INBOX")
        .arg("--limit")
        .arg(limit.to_string())
        .arg("--emit-interrupts")
        .arg(if baseline_only { "false" } else { "true" })
        .env("CTO_EMAIL_ADDRESS", &mail_env.address)
        .env("CTO_EMAIL_PASSWORD", &mail_env.password)
        .env("CTO_AGENT_BINARY", &mail_tooling.agent_binary_path)
        .current_dir(&paths.root)
        .output()
        .with_context(|| {
            format!(
                "failed to run periodic email sync through {}",
                path_display_name(&mail_tooling.script_path)
            )
        })?;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let payload = parse_json_command_output(&stdout)
        .ok_or_else(|| anyhow::anyhow!(compact_command_failure_note(&stdout, &stderr)))?;
    if !output.status.success() {
        let fallback = compact_command_failure_note(&stdout, &stderr);
        let error = payload
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or(fallback.as_str())
            .to_string();
        anyhow::bail!("Periodic email sync failed: {error}");
    }
    let ok_flag = payload.get("ok").and_then(Value::as_bool).unwrap_or(false);
    if !ok_flag {
        let fallback = compact_command_failure_note(&stdout, &stderr);
        let error = payload
            .get("error")
            .and_then(Value::as_str)
            .unwrap_or(fallback.as_str())
            .to_string();
        anyhow::bail!("Periodic email sync returned without ok=true: {error}");
    }
    let fetched_count = payload
        .get("fetchedCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    let stored_count = payload
        .get("storedCount")
        .and_then(Value::as_i64)
        .unwrap_or(0);
    Ok(MailSyncOutcome {
        fetched_count,
        stored_count,
        note: if baseline_only {
            format!(
                "Initial inbox baseline sync fetched {} messages and stored {} historical records through {} without emitting interrupts.",
                fetched_count,
                stored_count,
                path_display_name(&mail_tooling.script_path)
            )
        } else {
            format!(
                "Periodic inbox sync fetched {} messages and stored {} through {}.",
                fetched_count,
                stored_count,
                path_display_name(&mail_tooling.script_path)
            )
        },
    })
}

fn parse_runtime_env_text(text: &str) -> BTreeMap<String, String> {
    let mut env_map = BTreeMap::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let Some((key, raw_value)) = trimmed.split_once('=') else {
            continue;
        };
        env_map.insert(
            key.trim().to_string(),
            unquote_runtime_env_value(raw_value.trim()),
        );
    }
    env_map
}

fn unquote_runtime_env_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 {
        let bytes = trimmed.as_bytes();
        if (bytes[0] == b'"' && bytes[trimmed.len() - 1] == b'"')
            || (bytes[0] == b'\'' && bytes[trimmed.len() - 1] == b'\'')
        {
            return trimmed[1..trimmed.len() - 1].to_string();
        }
    }
    trimmed.to_string()
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

fn concise_reply_line(text: &str) -> String {
    let line = text
        .lines()
        .map(str::trim)
        .find(|line| !line.is_empty())
        .unwrap_or_default()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    if line.chars().count() <= 180 {
        return line;
    }
    let truncated = line.chars().take(177).collect::<String>();
    format!("{truncated}...")
}

fn derive_interrupt_reply_text(
    output_text: &str,
    checkpoint_summary: &str,
    checkpoint_detail: &str,
) -> String {
    let candidate = if !output_text.trim().is_empty() {
        output_text.trim()
    } else if !checkpoint_summary.trim().is_empty() {
        checkpoint_summary.trim()
    } else {
        checkpoint_detail.trim()
    };
    candidate.chars().take(4000).collect()
}

fn owner_visible_boot_reply_text(reply_text: &str) -> Option<String> {
    let trimmed = reply_text.trim();
    if trimmed.is_empty() {
        return None;
    }
    if trimmed.starts_with('{') || trimmed.starts_with('[') {
        return None;
    }
    let lowered = trimmed.to_ascii_lowercase();
    if lowered.contains("\"taskstatus\"")
        || lowered.contains("\"nextmode\"")
        || lowered.contains("\"checkpointsummary\"")
        || lowered.contains("the kleinhirn endpoint responded with unusable output")
        || lowered.contains("reclassify the task instead")
        || lowered.contains("current agent mode: mode=")
        || lowered.contains("workspace execution contract is active")
        || lowered.contains("all systems nominal. awaiting next command.")
        || lowered.contains("i am currently unable to produce any new progress")
    {
        return None;
    }
    Some(trimmed.to_string())
}

fn owner_interrupt_demands_verified_host_action(task: &crate::runtime_db::TaskRecord) -> bool {
    let lowered = format!("{} {}", task.title, task.detail).to_ascii_lowercase();
    [
        "keyboard",
        "tastatur",
        "layout",
        "deutsch",
        "german",
        "de/de-latin1",
        "x11",
        "wayland",
        "kde",
        "systemeinstellung",
        "localectl",
        "setxkbmap",
        "xkb",
    ]
    .iter()
    .any(|needle| lowered.contains(needle))
}

fn owner_interrupt_can_complete_directly(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    task_status: &str,
    next_mode: &str,
    used_machine_path: bool,
    output_text: &str,
) -> bool {
    if task.task_kind != "owner_interrupt"
        || task.source_interrupt_id.is_none()
        || task_status == "blocked"
        || matches!(
            next_mode,
            "delegate" | "blocked" | "await_review" | "request_resources"
        )
        || used_machine_path
        || workspace_execution_contract_active(paths, task.id)
        || owner_interrupt_demands_verified_host_action(task)
    {
        return false;
    }

    owner_visible_boot_reply_text(output_text).is_some()
}

fn publish_task_result_to_origin(
    paths: &Paths,
    task: &crate::runtime_db::TaskRecord,
    task_status: &str,
    checkpoint_summary: &str,
    checkpoint_detail: &str,
    output_text: &str,
    task_used_grosshirn: bool,
) {
    let reply_text =
        derive_interrupt_reply_text(output_text, checkpoint_summary, checkpoint_detail);
    if reply_text.trim().is_empty() {
        return;
    }

    if let Some(interrupt_id) = task
        .source_interrupt_id
        .filter(|_| task_status != "continue")
    {
        let _ = crate::runtime_db::complete_loop_interrupt(paths, interrupt_id, &reply_text);
        let payload = serde_json::json!({
            "interruptId": interrupt_id,
            "sourceChannel": task.source_channel,
            "speaker": task.speaker,
            "taskStatus": task_status,
            "reply": reply_text,
        });
        let _ = crate::runtime_db::record_agent_event(
            paths,
            "interrupt/replied",
            Some(task.id),
            &task.title,
            &concise_reply_line(&reply_text),
            &payload.to_string(),
        );
    }

    if matches!(task.source_channel.as_str(), "terminal" | "attach_terminal") {
        if let Some(visible_reply) = owner_visible_boot_reply_text(&reply_text) {
            let _ = append_boot_entry(paths, "cto-agent", &visible_reply);
        }
    }

    if task.source_channel == "bios" || is_owner_facing_task(task) {
        let _ = record_bios_dialogue(paths, "cto-agent", &reply_text, task_used_grosshirn);
    }
}

fn is_owner_facing_task(task: &crate::runtime_db::TaskRecord) -> bool {
    matches!(
        task.task_kind.as_str(),
        "owner_interrupt"
            | "root_trust"
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
    if list_open_tasks(paths, 16)?
        .into_iter()
        .any(|task| task.task_kind == "owner_interrupt")
    {
        return Ok(());
    }

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
            "Secure homepage and BIOS as a real communication bridge",
            "The visible homepage or BIOS bridge is not robust enough yet. Make sure homepage and BIOS really work, then schedule a mandatory browser self-review afterward.",
            880,
        )?;
    }

    if !root_auth.configured {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "root_trust",
            "Actively guide the owner to the superpassword",
            "The CTO-Agent must not merely claim root trust. Ask actively for superpassword setup through the root-auth page and explain why that root binding is necessary for BIOS, emergencies, and constitutional questions.",
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
            "Request owner, organigram, and product responsibility",
            "The organigram is not reliable yet. Ask about the owner, CEO or board, reports-to chain, peer CXOs, subordinate people, agents, vendors, and directly about the product or service for which the CTO-Agent should carry technical responsibility.",
            890,
        )?;
    }

    if !trust.owner_contact_established || !trust.bios_primary_channel_confirmed {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "owner_binding",
            "Pull owner communication into the BIOS path",
            "The agent does not have a reliable owner binding yet. Guide the owner into BIOS chat, ask about product, service, and technical priority, and consciously adopt that channel as the primary trust path.",
            875,
        )?;
    }

    let owner_email_assigned =
        !organigram.owner.email.trim().is_empty() || !bios.owner.email.trim().is_empty();
    let mail_runtime_configured = load_runtime_mail_env(paths).is_ok();
    let mail_reply_path_ready = ensure_mail_reply_path_ready(paths).is_ok();
    let email_sync_incident_open =
        has_open_loop_incident(paths, "email_sync_unavailable").unwrap_or(false);
    if (owner_email_assigned || mail_runtime_configured)
        && (!mail_reply_path_ready || email_sync_incident_open)
    {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "communication_governance",
            "Keep assigned email communication bidirectional",
            "An assigned or configured email path exists, so the CTO-Agent must keep it alive in both directions. Verify runtime credentials, the JS mail adapter, periodic inbox sync, and the interrupt bridge. Do not allow outbound email capability to drift away from inbound owner-reply capability.",
            868,
        )?;
    }

    if !bios.frozen {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "bios_freeze",
            "Prepare BIOS freeze actively",
            "The BIOS is not frozen yet. First verify superpassword, owner binding, organigram, and the visible BIOS page. Only then may BIOS freeze count as completed.",
            860,
        )?;
    }

    if local_kleinhirn_upgrade_available(paths) {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "model_or_resource",
            "Upgrade the local kleinhirn or reject it with justification",
            "The host already carries a stronger local kleinhirn than the current runtime. Evaluate the local upgrade seriously, execute it, or document clearly why it must not happen yet.",
            840,
        )?;
    }

    let grosshirn_configured = grosshirn_runtime_configured(paths);
    if trust.brain_access_mode != "kleinhirn_plus_grosshirn" && grosshirn_configured {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "grosshirn_activation",
            "Activate and verify grosshirn mode",
            "Grosshirn credentials or runtime configuration already exist, but brain access is not set to `kleinhirn_plus_grosshirn` yet. Decide cost-consciously whether you really need temporary grosshirn for this situation now, activate it through its own `brainAction` step if needed, and then verify local kleinhirn fallback.",
            830,
        )?;
    }
    if trust.brain_access_mode != "kleinhirn_plus_grosshirn" && !grosshirn_configured {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "grosshirn_procurement",
            "Prepare grosshirn procurement and owner approval",
            "The agent is continuing with only local kleinhirn for now. Research which grosshirn would be the best next step for coding, debugging, and complex CTO work, and prepare a concrete owner request with upside, cost, and approval path.",
            720,
        )?;
    }

    if focus.as_ref().map(|state| state.queue_depth).unwrap_or(0) == 0 {
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "environment_discovery",
            "Actively explore the environment and map blind spots",
            "There is currently no acutely prioritized external work. Do not treat this state as sleep, but as a CTO exploration phase. Inspect the host, running services, file surfaces, runtime limits, communication paths, and unknown risks in read-only mode. Record concretely what you learned about this environment, what remains unclear, and which new work should arise from it.",
            260,
        )?;
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "tool_exploration",
            "Test internal tools in a controlled way and document boundaries",
            "There is currently no acutely prioritized external work. Therefore test the available tool paths in a controlled and preferably read-only way: browser, bounded exec, exec session, homepage/BIOS routes, census, and diagnostic paths. Do not invent tool competence. Document for each meaningful tool what it is good for, what it is not good for, where the risks are, and which tool contracts should later be sharpened.",
            250,
        )?;
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "progress_reflection",
            "Define improvement and extend the progress journal",
            "There is currently no acutely prioritized external work. Reflect explicitly on what improvement means in your concrete CTO context: better owner binding, better product understanding, better system visibility, better tool use, better browser capability, better resource posture, better reports, or better governance. Compare the current state with your previous progress journal and the active learning path, decide honestly whether real improvement happened, and name the next self-generated improvement tasks.",
            240,
        )?;
        let _ = crate::runtime_db::enqueue_internal_task(
            paths,
            None,
            "person_relationship_review",
            "Maintain people paths and derive helpful suggestions only as drafts",
            "There is currently no acutely prioritized external work. Use people paths, conversation notes, learning references, and existing mail traces so you do not forget people. If a genuinely helpful suggestion for exactly one person emerges from that, formulate at most a validation-required contact draft instead of claiming an unreviewed outreach.",
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

fn grounded_exec_command_summary(
    command: &[String],
    exit_code: Option<i32>,
    timed_out: bool,
) -> String {
    if timed_out {
        format!(
            "Executed single bounded command, but it timed out. Bounded command-exec executed: {:?}",
            command
        )
    } else if let Some(code) = exit_code {
        if code == 0 {
            format!(
                "Executed single bounded command successfully. Bounded command-exec executed: {:?}",
                command
            )
        } else {
            format!(
                "Executed single bounded command with exit code {}. Bounded command-exec executed: {:?}",
                code, command
            )
        }
    } else {
        format!(
            "Executed single bounded command. Bounded command-exec executed: {:?}",
            command
        )
    }
}

fn latest_context_package_value(paths: &Paths, task_id: i64) -> Option<serde_json::Value> {
    crate::runtime_db::latest_context_package_for_task(paths, task_id)
        .ok()
        .flatten()
        .and_then(|package| serde_json::from_str::<serde_json::Value>(&package.package_json).ok())
}

fn workspace_execution_contract_active(paths: &Paths, task_id: i64) -> bool {
    latest_context_package_value(paths, task_id)
        .and_then(|value| value.get("rawInclusions").cloned())
        .and_then(|value| value.as_array().cloned())
        .map(|items| {
            items.iter().any(|item| {
                item.get("sourceKind").and_then(serde_json::Value::as_str)
                    == Some("system_capability_contract")
                    && item
                        .get("sourceRef")
                        .and_then(serde_json::Value::as_str)
                        .map(|value| value.ends_with("workspace-execution-capability-policy.json"))
                        .unwrap_or(false)
            })
        })
        .unwrap_or(false)
}

fn grounded_workspace_non_machine_summary() -> String {
    "Workspace execution contract is active and no machine path ran in this turn, so persisted progress stays at planning or inspection only.".to_string()
}

fn grounded_workspace_non_machine_output_text(model_output: &str) -> String {
    let mut message = "Workspace execution contract is active and no exec/browser machine path ran in this turn, so this reply remains planning or inspection only. Any claimed exec session, code edit, build, test, commit, or runtime result is unverified until the matching machine directive actually runs.".to_string();
    if !model_output.trim().is_empty() {
        message.push_str("\n\nModel-declared reply before contract grounding:\n");
        message.push_str(model_output.trim());
    }
    message
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
                format!("Started Codex exec session {}.", session_id),
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
                format!("Wrote to Codex exec session {}.", session_id),
                format!(
                    "{}\nSession: {}\nSent input bytes: {}\nClose stdin: {}\nSnapshot:\n{}",
                    ack,
                    session_id,
                    directive
                        .input
                        .as_ref()
                        .map(|value| value.len())
                        .unwrap_or(0),
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
            Ok((format!("Read Codex exec session {}.", session_id), rendered))
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
                format!("Terminated Codex exec session {}.", session_id),
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
        format!("Reused Codex exec session {}.", session_id),
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
        vec!["/bin/bash".to_string(), "-lc".to_string(), parsed.join(" ")]
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

fn render_exec_session_snapshot(snapshot: Option<&crate::command_exec::SessionSnapshot>) -> String {
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
        cpu_threads: std::thread::available_parallelism()
            .ok()
            .map(|value| value.get()),
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
            let recommended_candidate =
                json.get("candidates")
                    .and_then(Value::as_array)
                    .and_then(|items| {
                        items.iter().find(|item| {
                            item.get("recommended").and_then(Value::as_bool) == Some(true)
                        })
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
        let bytes = String::from_utf8(output.stdout)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
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
    fn completion_review_requires_explicit_approve_to_finish_parent() {
        let (resolution, note) = resolve_completion_review("done", None);
        assert_eq!(resolution, CompletionReviewResolution::Revise);
        assert!(
            note.unwrap_or_default()
                .contains("did not explicitly approve")
        );

        let review = crate::agentic::CompletionReviewDirective {
            decision: "approve".to_string(),
            note: "Looks good.".to_string(),
            evidence_gaps: Vec::new(),
            confidence: Some(0.9),
        };
        let (resolution, note) = resolve_completion_review("continue", Some(&review));
        assert_eq!(resolution, CompletionReviewResolution::Approve);
        assert!(note.is_none());
    }

    #[test]
    fn review_wait_recovery_has_safe_default_interval() {
        let _guard = test_lock().lock().expect("test lock poisoned");
        unsafe {
            std::env::remove_var("CTO_AGENT_REVIEW_WAIT_RECOVERY_INTERVAL_SECS");
        }
        assert_eq!(review_wait_recovery_interval(), Duration::from_secs(30));
        assert!(should_start_review_wait_recovery(
            None,
            review_wait_recovery_interval()
        ));
        assert!(!should_start_review_wait_recovery(
            Some(Instant::now()),
            review_wait_recovery_interval()
        ));
    }

    #[test]
    fn grounded_workspace_non_machine_output_mentions_unverified_exec_sessions() {
        let text = grounded_workspace_non_machine_output_text(
            "I have opened a new interactive shell session in the project root. The session ID is `cxx-01`.",
        );
        assert!(text.contains("Any claimed exec session"));
        assert!(text.contains("Model-declared reply before contract grounding"));
        assert!(text.contains("cxx-01"));
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
    fn open_owner_interrupt_suppresses_obligation_backlog() {
        with_temp_runtime("owner-interrupt-suppresses-obligations", |paths| {
            crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "owner_interrupt",
                "Single owner mission",
                "Stay on the owner's direct mission and do not queue governance side work.",
                1000,
            )
            .expect("owner interrupt should enqueue");

            ensure_cto_obligations(paths).expect("obligation generation should succeed");

            let queued = list_queued_tasks(paths, 64).expect("queued tasks should load");
            assert_eq!(queued.len(), 1);
            assert_eq!(queued[0].task_kind, "owner_interrupt");
        });
    }

    #[test]
    fn orphaned_active_turn_is_recovered_back_into_queue() {
        with_temp_runtime("orphaned-turn-recovery", |paths| {
            crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "tool_exploration",
                "Tool test",
                "Temporary test task for orphan recovery.",
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
            let thread =
                crate::runtime_db::load_agent_thread(paths).expect("agent thread should reload");
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
                "Tool test",
                "Temporary test task for live watchdog recovery.",
                250,
            )
            .expect("should enqueue internal task");
            let task = crate::runtime_db::select_next_task(paths)
                .expect("should select next task")
                .expect("selected task should exist");
            let turn = start_agent_turn(paths, task.id, &task.title, "test", "execute_task")
                .expect("should create turn");

            let recovered =
                watchdog_interrupt_live_turn(paths, turn.id, task.id, &task.title, 901, 300)
                    .expect("watchdog recovery should succeed");
            assert!(recovered.is_some());
            assert!(!is_agent_turn_in_progress(paths, turn.id).expect("turn state should load"));
            let recovered_task = crate::runtime_db::load_task_by_id(paths, task.id)
                .expect("task should reload")
                .expect("task should still exist");
            assert_eq!(recovered_task.status, "queued");
            let queued = list_queued_tasks(paths, 64).expect("queued tasks should load");
            assert!(queued.iter().any(|task| task.task_kind == "recovery"));
        });
    }

    #[test]
    fn interrupt_signal_preempts_live_turn_for_higher_priority_interrupt_task() {
        with_temp_runtime("interrupt-preemption", |paths| {
            crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "model_or_resource",
                "Handle the kleinhirn/resource question",
                "Active test task for interrupt preemption.",
                1000,
            )
            .expect("should enqueue primary task");
            let active_task = crate::runtime_db::select_next_task(paths)
                .expect("should select next task")
                .expect("active task should exist");
            let turn = start_agent_turn(
                paths,
                active_task.id,
                &active_task.title,
                "test",
                "execute_task",
            )
            .expect("should create turn");

            let interrupt_id = crate::runtime_db::enqueue_loop_interrupt(
                paths,
                "attach_terminal",
                "Michael Welsch",
                "Stop the current meta loop and answer the owner now.",
            )
            .expect("interrupt should enqueue");
            let queued_task = crate::runtime_db::queue_loop_interrupt_as_task(paths, interrupt_id)
                .expect("interrupt task should queue")
                .expect("queued task should exist");
            let signal = crate::runtime_db::record_turn_signal_for_active_turn(
                paths,
                "attach_terminal",
                "Michael Welsch",
                "Stop the current meta loop and answer the owner now.",
            )
            .expect("signal should record")
            .expect("signal should exist");

            let candidate = find_interrupt_preemption_candidate(paths, &active_task, turn.id)
                .expect("candidate lookup should succeed")
                .expect("candidate should exist");
            assert_eq!(candidate.0.id, queued_task.id);
            assert_eq!(candidate.1.id, signal.id);

            let summary = crate::runtime_db::interrupt_live_turn_for_signal_preemption(
                paths,
                turn.id,
                active_task.id,
                &active_task.title,
                5,
                queued_task.id,
                &queued_task.title,
                &signal,
            )
            .expect("preemption should succeed");
            assert!(summary.is_some());

            let refreshed_turn = crate::runtime_db::list_recent_agent_turns(paths, 4)
                .expect("turns should reload")
                .into_iter()
                .find(|entry| entry.id == turn.id)
                .expect("turn should exist");
            assert_eq!(refreshed_turn.status, "interrupted_by_signal");

            let refreshed_task = crate::runtime_db::load_task_by_id(paths, active_task.id)
                .expect("task should reload")
                .expect("task should exist");
            assert_eq!(refreshed_task.status, "queued");

            let refreshed_signal = crate::runtime_db::list_recent_turn_signals(paths, 8)
                .expect("signals should reload")
                .into_iter()
                .find(|entry| entry.id == signal.id)
                .expect("signal should exist");
            assert_eq!(refreshed_signal.status, "consumed");

            let focus = crate::runtime_db::load_focus_state(paths).expect("focus should load");
            assert_eq!(focus.mode, "reprioritize");
            assert_eq!(focus.active_task_id, Some(queued_task.id));
        });
    }

    #[test]
    fn owner_interrupt_preempts_equal_priority_self_preservation_task() {
        with_temp_runtime("owner-interrupt-equal-priority-preemption", |paths| {
            crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "self_preservation",
                "Secure Infinity Loop self-preservation",
                "Active self-preservation task should yield to a fresh owner interrupt.",
                1000,
            )
            .expect("should enqueue self-preservation task");
            let active_task = crate::runtime_db::select_next_task(paths)
                .expect("should select next task")
                .expect("active task should exist");
            assert_eq!(active_task.task_kind, "self_preservation");
            assert_eq!(active_task.priority_score, 1000);

            let turn = start_agent_turn(
                paths,
                active_task.id,
                &active_task.title,
                "test",
                "execute_task",
            )
            .expect("should create turn");

            let interrupt_id = crate::runtime_db::enqueue_loop_interrupt(
                paths,
                "attach_terminal",
                "Michael Welsch",
                "Stop the self-preservation meta and answer the owner now.",
            )
            .expect("interrupt should enqueue");
            let queued_task = crate::runtime_db::queue_loop_interrupt_as_task(paths, interrupt_id)
                .expect("interrupt task should queue")
                .expect("queued task should exist");
            assert_eq!(queued_task.task_kind, "owner_interrupt");
            assert_eq!(queued_task.priority_score, active_task.priority_score);

            let signal = crate::runtime_db::record_turn_signal_for_active_turn(
                paths,
                "attach_terminal",
                "Michael Welsch",
                "Stop the self-preservation meta and answer the owner now.",
            )
            .expect("signal should record")
            .expect("signal should exist");

            let candidate = find_interrupt_preemption_candidate(paths, &active_task, turn.id)
                .expect("candidate lookup should succeed")
                .expect("candidate should exist");
            assert_eq!(candidate.0.id, queued_task.id);
            assert_eq!(candidate.1.id, signal.id);
        });
    }

    #[test]
    fn sticky_continue_yields_to_queued_owner_interrupt_at_boundary() {
        with_temp_runtime("sticky-continue-owner-boundary-preemption", |paths| {
            crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "root_trust",
                "Handle root trust follow-up",
                "A non-owner task is currently active.",
                120,
            )
            .expect("should enqueue root trust task");
            let active_task = crate::runtime_db::select_next_task(paths)
                .expect("should select next task")
                .expect("active task should exist");
            assert_eq!(active_task.task_kind, "root_trust");

            let interrupt_id = crate::runtime_db::enqueue_loop_interrupt(
                paths,
                "bios",
                "Michael Welsch",
                "Drop everything and handle my BIOS owner interrupt now.",
            )
            .expect("interrupt should enqueue");
            let queued_owner_task =
                crate::runtime_db::queue_loop_interrupt_as_task(paths, interrupt_id)
                    .expect("interrupt task should queue")
                    .expect("queued task should exist");
            assert_eq!(queued_owner_task.task_kind, "owner_interrupt");

            let mut next_mode = "execute_task".to_string();
            let checkpoint_summary =
                "The active task made progress but is still mid-work and would normally continue.";
            let mut checkpoint_detail =
                "Verified state changed and the task would otherwise stay active.".to_string();

            persist_task_progress_with_boundary_preemption(
                paths,
                &active_task,
                &mut next_mode,
                checkpoint_summary,
                &mut checkpoint_detail,
                Some("progress"),
            )
            .expect("persist should succeed");

            assert_eq!(next_mode, "reprioritize");
            assert!(checkpoint_detail.contains("Boundary preemption:"));
            assert!(checkpoint_detail.contains(&queued_owner_task.id.to_string()));

            let refreshed_active = crate::runtime_db::load_task_by_id(paths, active_task.id)
                .expect("task reload should succeed")
                .expect("active task should still exist");
            assert_eq!(refreshed_active.status, "queued");
            assert_eq!(
                refreshed_active.last_checkpoint_summary.as_deref(),
                Some(checkpoint_summary)
            );

            let focus = crate::runtime_db::load_focus_state(paths).expect("focus should load");
            assert_eq!(focus.mode, "reprioritize");

            let checkpoints = crate::runtime_db::list_task_checkpoints(paths, active_task.id, 4)
                .expect("checkpoints should load");
            assert!(
                checkpoints
                    .iter()
                    .any(|entry| entry.detail.contains("Boundary preemption:"))
            );
        });
    }

    #[test]
    fn sticky_continue_yields_to_queued_owner_email_interrupt_at_boundary() {
        with_temp_runtime("sticky-continue-owner-email-boundary-preemption", |paths| {
            crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "owner_interrupt",
                "Send the owner a test mail",
                "The active owner task is a visible mail step.",
                1000,
            )
            .expect("should enqueue owner task");
            let active_task = crate::runtime_db::select_next_task(paths)
                .expect("should select next task")
                .expect("active task should exist");
            assert_eq!(active_task.task_kind, "owner_interrupt");

            let conn =
                rusqlite::Connection::open(&paths.runtime_db_path).expect("runtime db should open");
            conn.execute(
                "INSERT INTO tasks(
                    created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                    title, detail, trust_level, priority_score, status
                 ) VALUES(?1, ?2, NULL, NULL, ?3, 'email', 'Michael Welsch <michael.welsch@metric-space.ai>', 'owner_interrupt',
                    'Owner email interrupt', 'Stop sending the same mail again and again.', 'owner', 1000, 'queued')",
                rusqlite::params![now_iso(), now_iso(), 9001],
            )
            .expect("queued owner email task should insert");
            let queued_owner_task =
                crate::runtime_db::load_task_by_id(paths, conn.last_insert_rowid())
                    .expect("queued task should load")
                    .expect("queued task should exist");
            assert_eq!(queued_owner_task.task_kind, "owner_interrupt");
            assert_eq!(queued_owner_task.source_channel, "email");

            let mut next_mode = "execute_task".to_string();
            let checkpoint_summary =
                "Visible owner mail was sent, but the task would otherwise stay active.";
            let mut checkpoint_detail =
                "The current owner task would normally continue after a bounded step.".to_string();

            persist_task_progress_with_boundary_preemption(
                paths,
                &active_task,
                &mut next_mode,
                checkpoint_summary,
                &mut checkpoint_detail,
                Some("progress"),
            )
            .expect("persist should succeed");

            assert_eq!(next_mode, "reprioritize");
            assert!(checkpoint_detail.contains("preempts task"));

            let refreshed_active = crate::runtime_db::load_task_by_id(paths, active_task.id)
                .expect("active task should reload")
                .expect("active task should still exist");
            assert_eq!(refreshed_active.status, "queued");

            let focus = crate::runtime_db::load_focus_state(paths).expect("focus should load");
            assert_eq!(focus.mode, "reprioritize");
        });
    }

    #[test]
    fn sticky_continue_ignores_older_owner_interrupt_at_boundary() {
        with_temp_runtime("sticky-continue-older-owner-does-not-preempt", |paths| {
            let older_interrupt = crate::runtime_db::enqueue_loop_interrupt(
                paths,
                "bios",
                "Michael Welsch",
                "Older owner interrupt.",
            )
            .expect("older interrupt should enqueue");
            let older_task =
                crate::runtime_db::queue_loop_interrupt_as_task(paths, older_interrupt)
                    .expect("older interrupt task should queue")
                    .expect("older queued task should exist");

            let newer_interrupt = crate::runtime_db::enqueue_loop_interrupt(
                paths,
                "attach_terminal",
                "Michael Welsch",
                "Newer owner interrupt.",
            )
            .expect("newer interrupt should enqueue");
            let newer_task =
                crate::runtime_db::queue_loop_interrupt_as_task(paths, newer_interrupt)
                    .expect("newer interrupt task should queue")
                    .expect("newer queued task should exist");

            let active_task = crate::runtime_db::activate_selected_task(paths, newer_task.id)
                .expect("newer task should activate")
                .expect("active task should exist");
            assert_eq!(active_task.id, newer_task.id);

            let mut next_mode = "execute_task".to_string();
            let checkpoint_summary =
                "The newer owner task made progress and would normally continue.";
            let mut checkpoint_detail =
                "Fresh machine evidence exists for the currently active owner task.".to_string();

            persist_task_progress_with_boundary_preemption(
                paths,
                &active_task,
                &mut next_mode,
                checkpoint_summary,
                &mut checkpoint_detail,
                Some("progress"),
            )
            .expect("persist should succeed");

            assert_eq!(next_mode, "execute_task");
            assert!(!checkpoint_detail.contains("Boundary preemption:"));

            let refreshed_active = crate::runtime_db::load_task_by_id(paths, active_task.id)
                .expect("newer task should reload")
                .expect("newer task should still exist");
            assert_eq!(refreshed_active.status, "active");
            assert_eq!(older_task.status, "queued");
        });
    }

    #[test]
    fn reviewed_js_owner_mail_send_counts_as_completion() {
        with_temp_runtime("reviewed-js-owner-mail-completion", |paths| {
            let conn =
                rusqlite::Connection::open(&paths.runtime_db_path).expect("runtime db should open");
            conn.execute_batch(
                "CREATE TABLE communication_messages (
                    message_key TEXT PRIMARY KEY,
                    channel TEXT NOT NULL,
                    account_key TEXT NOT NULL,
                    thread_key TEXT NOT NULL,
                    remote_id TEXT NOT NULL,
                    direction TEXT NOT NULL,
                    folder_hint TEXT NOT NULL,
                    sender_display TEXT NOT NULL,
                    sender_address TEXT NOT NULL,
                    recipient_addresses_json TEXT NOT NULL,
                    cc_addresses_json TEXT NOT NULL,
                    bcc_addresses_json TEXT NOT NULL,
                    subject TEXT NOT NULL,
                    preview TEXT NOT NULL,
                    body_text TEXT NOT NULL,
                    body_html TEXT NOT NULL,
                    raw_payload_ref TEXT NOT NULL,
                    trust_level TEXT NOT NULL,
                    status TEXT NOT NULL,
                    seen INTEGER NOT NULL,
                    has_attachments INTEGER NOT NULL,
                    external_created_at TEXT NOT NULL,
                    observed_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                );",
            )
            .expect("communication schema should exist");
            conn.execute(
                "INSERT INTO communication_messages(
                    message_key, channel, account_key, thread_key, remote_id, direction, folder_hint,
                    sender_display, sender_address, recipient_addresses_json, cc_addresses_json,
                    bcc_addresses_json, subject, preview, body_text, body_html, raw_payload_ref,
                    trust_level, status, seen, has_attachments, external_created_at, observed_at,
                    metadata_json
                ) VALUES(
                    ?1, 'email', 'email:cto1@metric-space.ai', ?2, ?3, 'outbound', 'sent',
                    '', 'cto1@metric-space.ai', ?4, '[]', '[]', 'Status', 'Alles gut', 'Alles gut',
                    '', '', 'low', 'sent', 1, 0, ?5, ?5, ?6
                )",
                rusqlite::params![
                    "email:cto1@metric-space.ai::sent::<msg-1@example.test>",
                    "<msg-1@example.test>",
                    "<msg-1@example.test>",
                    "[\"michael.welsch@metric-space.ai\"]",
                    now_iso(),
                    "{\"messageId\":\"<msg-1@example.test>\"}"
                ],
            )
            .expect("outbound message should insert");

            let task = crate::runtime_db::TaskRecord {
                id: 75,
                created_at: now_iso(),
                updated_at: now_iso(),
                parent_task_id: None,
                worker_job_id: None,
                source_interrupt_id: Some(55),
                source_channel: "bios".to_string(),
                speaker: "Michael Welsch".to_string(),
                task_kind: "owner_interrupt".to_string(),
                title: "Schreibe mir eine Test-E-Mail".to_string(),
                detail: "Sende genau eine sichtbare Testmail.".to_string(),
                trust_level: "owner".to_string(),
                priority_score: 1000,
                status: "active".to_string(),
                run_count: 1,
                last_checkpoint_summary: None,
                last_checkpoint_at: None,
                last_output: None,
            };
            let directive = crate::tooling::ExecCommandDirective {
                command: vec![
                    "env".to_string(),
                    "CTO_EMAIL_ADDRESS=cto1@metric-space.ai".to_string(),
                    "CTO_EMAIL_PASSWORD=secret".to_string(),
                    "node".to_string(),
                    "scripts/communication_mail_cli.mjs".to_string(),
                    "--db".to_string(),
                    "runtime/cto_agent.db".to_string(),
                    "send".to_string(),
                    "--to".to_string(),
                    "michael.welsch@metric-space.ai".to_string(),
                    "--subject".to_string(),
                    "Status".to_string(),
                    "--body".to_string(),
                    "Alles gut".to_string(),
                ],
                workdir: None,
                timeout_ms: None,
                justification: Some("Send the requested visible owner mail.".to_string()),
            };
            let exec_result = crate::tooling::ExecCommandResult {
                status: "ok".to_string(),
                exit_code: Some(0),
                timed_out: false,
                stdout: "{\n  \"ok\": true,\n  \"message_id\": \"<msg-1@example.test>\"\n}"
                    .to_string(),
                stderr: String::new(),
                cwd: "/tmp".to_string(),
            };

            let completion =
                visible_owner_mail_send_completion(paths, &task, &directive, &exec_result)
                    .expect("mail send should satisfy completion");
            assert!(completion.0.contains("sent successfully"));
            assert!(completion.1.contains("<msg-1@example.test>"));
            assert!(completion.1.contains("michael.welsch@metric-space.ai"));
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
                title: "Handle the kleinhirn/resource question".to_string(),
                detail: "Check the local kleinhirn upgrade.".to_string(),
                trust_level: "owner".to_string(),
                priority_score: 860,
                status: "active".to_string(),
                run_count: 2,
                last_checkpoint_summary: Some(
                    "Model returned empty text; bounded retry instead of hard block.".to_string(),
                ),
                last_checkpoint_at: Some(now_iso()),
                last_output: None,
            };

            let escalation = assess_task_stuck_risk(
                &paths,
                &task,
                &loop_safety,
                "newborn",
                "Model returned empty text; bounded retry instead of hard block.",
                "Kleinhirn returned no usable text.",
                "continue",
                "reprioritize",
            )
            .expect("stuck-risk escalation should exist");

            assert_eq!(escalation.task_status, "continue");
            assert_eq!(escalation.next_mode, "review");
            assert!(
                escalation
                    .checkpoint_summary
                    .contains("Local kleinhirn upgrade stays on the local review path")
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
            title: "Handle the kleinhirn/resource question".to_string(),
            detail: "Check honestly now whether you can upgrade your local kleinhirn to Qwen3.5-35B-A3B.".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "active".to_string(),
            run_count: 28,
            last_checkpoint_summary: Some(
                "Task 338 will not be continued blindly. The task has used too many bounded attempts for the current maturity stage. Maturity stage: newborn Local kleinhirn upgrade stays on the local review path."
                    .to_string(),
            ),
            last_checkpoint_at: Some(now_iso()),
            last_output: None,
        };

        assert!(should_force_local_kleinhirn_self_repair(
            &task,
            "done",
            "review",
            "Task 338 will not be continued blindly. The task has used too many bounded attempts for the current maturity stage. Maturity stage: newborn Local kleinhirn upgrade stays on the local review path.",
            "Local review is spinning in circles.",
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
            title: "Handle the kleinhirn/resource question".to_string(),
            detail: "Check honestly now whether you can upgrade your local kleinhirn to Qwen3.5-35B-A3B.".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "active".to_string(),
            run_count: 28,
            last_checkpoint_summary: Some(
                "Task 338 will not be continued blindly. The task has used too many bounded attempts for the current maturity stage. Maturity stage: newborn Local kleinhirn upgrade stays on the local review path."
                    .to_string(),
            ),
            last_checkpoint_at: Some(now_iso()),
            last_output: None,
        };

        assert!(!should_force_local_kleinhirn_self_repair(
            &task,
            "continue",
            "review",
            "Task 338 will not be continued blindly. The task has used too many bounded attempts for the current maturity stage. Maturity stage: newborn Local kleinhirn upgrade stays on the local review path.",
            "Local review is spinning in circles.",
            false,
        ));
    }

    #[test]
    fn repeated_bounded_owner_interrupt_exec_stays_in_review_instead_of_requesting_resources() {
        with_temp_runtime("owner-interrupt-bounded-review", |paths| {
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
                    "Executed single bounded command to list repository root contents. Bounded command-exec executed: [\"ls\", \"-1\"]"
                        .to_string(),
                ),
                last_checkpoint_at: Some(now_iso()),
                last_output: Some(
                    "Executed `ls -1` in repository root. Output: README.md".to_string(),
                ),
            };

            let escalation = assess_task_stuck_risk(
                paths,
                &task,
                &loop_safety,
                "adaptive",
                "Executed single bounded command to list repository root contents. Bounded command-exec executed: [\"ls\", \"-1\"]",
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
                    .contains("The same bounded machine action")
                    || escalation
                        .checkpoint_summary
                        .contains("Dieselbe bounded Maschinenaktion")
            );
        });
    }

    #[test]
    fn repeated_exec_only_becomes_visible_after_final_checkpoint_mutation() {
        with_temp_runtime("owner-interrupt-post-exec-recheck", |paths| {
            let loop_safety = default_loop_safety_policy();
            let persisted_summary = "Executed single bounded command successfully. Bounded command-exec executed: [\"bash\", \"-lc\", \"cat <<'CPP' > src/main.cpp\"]";
            let task = crate::runtime_db::TaskRecord {
                id: 444,
                created_at: now_iso(),
                updated_at: now_iso(),
                parent_task_id: None,
                worker_job_id: None,
                source_interrupt_id: Some(11),
                source_channel: "attach_terminal".to_string(),
                speaker: "Michael Welsch".to_string(),
                task_kind: "owner_interrupt".to_string(),
                title: "Build the C++ console app".to_string(),
                detail: "Keep implementing the same owner-priority C++ workspace.".to_string(),
                trust_level: "system".to_string(),
                priority_score: 1000,
                status: "active".to_string(),
                run_count: 9,
                last_checkpoint_summary: Some(persisted_summary.to_string()),
                last_checkpoint_at: Some(now_iso()),
                last_output: None,
            };

            let pre_exec = assess_task_stuck_risk(
                paths,
                &task,
                &loop_safety,
                "newborn",
                "Continue implementing the C++ console application by adding the missing main function and basic command-line parsing.",
                "The model proposed another bounded command.",
                "continue",
                "execute_task",
            );
            assert!(
                pre_exec.is_none(),
                "the raw model summary alone does not yet reveal the repeated bounded exec"
            );

            let post_exec = assess_task_stuck_risk(
                paths,
                &task,
                &loop_safety,
                "newborn",
                persisted_summary,
                "Bounded exec result:\nCommand: [\"bash\", \"-lc\", \"cat <<'CPP' > src/main.cpp\"]",
                "continue",
                "reprioritize",
            )
            .expect("the final persisted checkpoint should trigger the repeat guard");
            assert_eq!(post_exec.task_status, "continue");
            assert_eq!(post_exec.next_mode, "review");
            assert!(
                post_exec
                    .checkpoint_summary
                    .contains("same bounded machine action")
                    || post_exec
                        .checkpoint_summary
                        .contains("same checkpoint without new movement")
            );
        });
    }

    #[test]
    fn grounded_exec_command_summary_uses_machine_outcome() {
        let command = vec!["bash".to_string(), "-lc".to_string(), "false".to_string()];
        let summary = grounded_exec_command_summary(&command, Some(1), false);
        assert!(summary.contains("exit code 1"));
        assert!(summary.contains("Bounded command-exec executed"));
        assert!(!summary.contains("Makefile"));
    }

    #[test]
    fn grounded_workspace_non_machine_summary_stays_planning_only() {
        let summary = grounded_workspace_non_machine_summary();
        assert!(summary.contains("planning or inspection only"));
        assert!(!summary.contains("compiled"));
        assert!(!summary.contains("Makefile"));
    }

    #[test]
    fn substantive_task_kind_supports_local_brain_switch_but_review_does_not() {
        assert!(task_kind_supports_task_local_brain_switch(
            "owner_interrupt"
        ));
        assert!(task_kind_supports_task_local_brain_switch("task"));
        assert!(!task_kind_supports_task_local_brain_switch("self_review"));
        assert!(!task_kind_supports_task_local_brain_switch("recovery"));
    }

    #[test]
    fn substantive_continue_defaults_back_into_execute_task() {
        let task = crate::runtime_db::TaskRecord {
            id: 12,
            created_at: now_iso(),
            updated_at: now_iso(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: Some(2),
            source_channel: "attach_terminal".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: "Finish the repo task".to_string(),
            detail: "Stay on the substantive owner task.".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "active".to_string(),
            run_count: 4,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        };

        assert_eq!(
            infer_next_mode(&task, execution_mode_for_task(&task), "continue"),
            "execute_task"
        );
        assert_eq!(
            normalize_sticky_continue_mode(&task, "continue", "reprioritize"),
            "execute_task"
        );
    }

    #[test]
    fn observe_mode_is_suppressed_while_a_task_workstream_is_active() {
        assert!(!should_enter_observe_mode(true, true));
        assert!(!should_enter_observe_mode(false, true));
        assert!(should_enter_observe_mode(false, false));
    }

    #[test]
    fn active_workstream_skips_reprioritize_and_ticks_agentic_every_cycle() {
        assert!(!should_run_reprioritize(true));
        assert!(should_run_reprioritize(false));
        assert!(should_tick_agentic_loop(2, true));
        assert!(should_tick_agentic_loop(3, true));
        assert!(should_tick_agentic_loop(1, false));
        assert!(should_tick_agentic_loop(2, false));
        assert!(!should_tick_agentic_loop(3, false));
    }

    #[test]
    fn kleinhirn_repair_task_starts_only_when_not_running_and_cooldown_elapsed() {
        let cooldown = Duration::from_secs(60);
        assert!(should_start_kleinhirn_repair_task(false, None, cooldown));
        assert!(!should_start_kleinhirn_repair_task(true, None, cooldown));
        assert!(!should_start_kleinhirn_repair_task(
            false,
            Some(Instant::now()),
            cooldown
        ));
        assert!(should_start_kleinhirn_repair_task(
            false,
            Some(Instant::now() - Duration::from_secs(61)),
            cooldown
        ));
    }

    #[test]
    fn owner_interrupt_run_count_limit_stays_in_review_instead_of_blocking() {
        with_temp_runtime("owner-interrupt-run-count-limit", |paths| {
            let loop_safety = default_loop_safety_policy();
            let task = crate::runtime_db::TaskRecord {
                id: 55,
                created_at: now_iso(),
                updated_at: now_iso(),
                parent_task_id: None,
                worker_job_id: None,
                source_interrupt_id: Some(8),
                source_channel: "attach_terminal".to_string(),
                speaker: "Michael Welsch".to_string(),
                task_kind: "owner_interrupt".to_string(),
                title: "Repo overview and CEO mail draft".to_string(),
                detail: "Read the owner briefing, installation bootstrap, and bounded repo evidence, then draft the CEO update.".to_string(),
                trust_level: "system".to_string(),
                priority_score: 1000,
                status: "active".to_string(),
                run_count: 3,
                last_checkpoint_summary: Some(
                    "Started one bounded evidence-gathering step: reading the owner briefing and repo structure."
                        .to_string(),
                ),
                last_checkpoint_at: Some(now_iso()),
                last_output: Some(
                    "I am verifying the required source artifacts first so the next turn can report exact verified facts.".to_string(),
                ),
            };

            let escalation = assess_task_stuck_risk(
                paths,
                &task,
                &loop_safety,
                "newborn",
                "Started one bounded evidence-gathering step: reading the installation bootstrap and ~/Downloads/I-hate-AI structure.",
                "Bounded exec result:\nVerified owner briefing, installation bootstrap, and repo path structure.",
                "continue",
                "reprioritize",
            );

            assert!(
                escalation.is_none(),
                "owner interrupt should keep working when each bounded step still moves forward"
            );
        });
    }

    #[test]
    fn owner_interrupt_still_escalates_on_repeated_checkpoint() {
        with_temp_runtime("owner-interrupt-repeat-review", |paths| {
            let loop_safety = default_loop_safety_policy();
            let task = crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "owner_interrupt",
                "Repo overview and CEO mail draft",
                "Read the owner briefing, installation bootstrap, and bounded repo evidence, then draft the CEO update.",
                1000,
            )
            .expect("owner task should enqueue");
            let repeated_summary = "Started one bounded evidence-gathering step: reading the owner briefing and repo structure.";
            crate::runtime_db::record_task_checkpoint(
                paths,
                task.id,
                "continue",
                repeated_summary,
                "First matching bounded evidence-gathering checkpoint.",
            )
            .expect("first matching checkpoint should persist");
            crate::runtime_db::record_task_checkpoint(
                paths,
                task.id,
                "continue",
                repeated_summary,
                "Second matching bounded evidence-gathering checkpoint.",
            )
            .expect("second matching checkpoint should persist");
            let mut task = crate::runtime_db::load_task_by_id(paths, task.id)
                .expect("task should reload")
                .expect("task should exist");
            task.status = "active".to_string();
            task.run_count = 5;
            task.last_checkpoint_summary = Some(repeated_summary.to_string());
            task.last_checkpoint_at = Some(now_iso());
            task.last_output = Some(
                "I am verifying the required source artifacts first so the next turn can report exact verified facts.".to_string(),
            );

            let escalation = assess_task_stuck_risk(
                paths,
                &task,
                &loop_safety,
                "newborn",
                repeated_summary,
                "Bounded exec result:\nVerified owner briefing, installation bootstrap, and repo path structure.",
                "continue",
                "reprioritize",
            )
            .expect("owner interrupt repeat escalation should exist");

            assert_eq!(escalation.task_status, "continue");
            assert_eq!(escalation.next_mode, "review");
            assert!(
                escalation
                    .checkpoint_summary
                    .contains("Owner interrupt stays on the direct owner-review path")
            );
            assert!(!escalation.spawn_self_preservation_task);
        });
    }

    #[test]
    fn owner_interrupt_single_repeated_checkpoint_does_not_cool_down_yet() {
        with_temp_runtime("owner-interrupt-single-repeat-no-cooldown", |paths| {
            let loop_safety = default_loop_safety_policy();
            let task = crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "owner_interrupt",
                "Stay on the owner mission",
                "Continue the same owner task without parking after a single same-summary repeat.",
                1000,
            )
            .expect("owner task should enqueue");
            let repeated_summary = "Started one bounded evidence-gathering step: reading the owner briefing and repo structure.";
            crate::runtime_db::record_task_checkpoint(
                paths,
                task.id,
                "continue",
                repeated_summary,
                "Only one prior matching checkpoint exists.",
            )
            .expect("matching checkpoint should persist");
            let mut task = crate::runtime_db::load_task_by_id(paths, task.id)
                .expect("task should reload")
                .expect("task should exist");
            task.status = "active".to_string();
            task.run_count = 5;
            task.last_checkpoint_summary = Some(repeated_summary.to_string());
            task.last_checkpoint_at = Some(now_iso());

            let escalation = assess_task_stuck_risk(
                paths,
                &task,
                &loop_safety,
                "newborn",
                repeated_summary,
                "Bounded exec result:\nVerified owner briefing, installation bootstrap, and repo path structure.",
                "continue",
                "reprioritize",
            );

            assert!(
                escalation.is_none(),
                "a single repeated summary should not yet park a substantive owner task"
            );
        });
    }

    #[test]
    fn substantive_owner_interrupt_does_not_escalate_from_run_count_alone_even_under_grosshirn() {
        with_temp_runtime("owner-interrupt-grosshirn-run-count-limit", |paths| {
            let loop_safety = default_loop_safety_policy();
            crate::runtime_db::set_brain_access_mode(paths, "kleinhirn_plus_grosshirn")
                .expect("brain mode should allow grosshirn");
            let task = crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "owner_interrupt",
                "Stay on the owner mission",
                "Continue the same owner task through temporary grosshirn until it either works or must cool down.",
                1000,
            )
            .expect("owner task should enqueue");
            crate::runtime_db::arm_task_grosshirn_boost(
                paths,
                task.id,
                &task.title,
                "test boost",
                1120,
            )
            .expect("grosshirn boost should arm");
            let mut active = crate::runtime_db::load_task_by_id(paths, task.id)
                .expect("task should reload")
                .expect("task should exist");
            active.status = "active".to_string();
            active.run_count = 4;
            active.last_checkpoint_summary = Some(
                "Previous bounded grosshirn step still stayed inside repo rediscovery.".to_string(),
            );
            active.last_checkpoint_at = Some(now_iso());
            active.last_output = Some(
                "The last bounded grosshirn step did not yet unlock implementation movement."
                    .to_string(),
            );

            let escalation = assess_task_stuck_risk(
                paths,
                &active,
                &loop_safety,
                "newborn",
                "Another bounded grosshirn step is still rediscovering the same repo surface.",
                "Bounded exec result:\nRecovered the same repo anchors again without advancing implementation.",
                "continue",
                "reprioritize",
            );

            assert!(
                escalation.is_none(),
                "substantive owner work should not be yanked out of the task loop by run count alone"
            );
        });
    }

    #[test]
    fn context_preparation_total_iteration_limit_blocks_at_four() {
        with_temp_runtime("context-preparation-total-iteration-limit", |paths| {
            let loop_safety = default_loop_safety_policy();
            let task = crate::runtime_db::TaskRecord {
                id: 145,
                created_at: now_iso(),
                updated_at: now_iso(),
                parent_task_id: Some(144),
                worker_job_id: None,
                source_interrupt_id: None,
                source_channel: "system_guard".to_string(),
                speaker: "system_guard".to_string(),
                task_kind: "context_preparation".to_string(),
                title: "Prepare context for task 144".to_string(),
                detail: "Rewrite the execution handoff for the current owner task.".to_string(),
                trust_level: "system".to_string(),
                priority_score: 1005,
                status: "active".to_string(),
                run_count: 4,
                last_checkpoint_summary: Some(
                    "Context optimization checkpoint: rewrite total iteration 4/4 for task #145."
                        .to_string(),
                ),
                last_checkpoint_at: Some(now_iso()),
                last_output: Some(
                    "The previous rewrite still missed the final verified world state block."
                        .to_string(),
                ),
            };

            let escalation = assess_task_stuck_risk(
                paths,
                &task,
                &loop_safety,
                "newborn",
                "Context optimization checkpoint: review total iteration 5/4 for task #145.",
                "The latest prepared context still needs another rewrite pass.",
                "continue",
                "reprioritize",
            )
            .expect("context preparation should stop at the total loop limit");

            assert_eq!(escalation.task_status, "blocked");
            assert_eq!(escalation.next_mode, "review");
            assert!(
                escalation
                    .checkpoint_summary
                    .contains("maximum total iteration budget")
            );
            assert!(!escalation.spawn_self_preservation_task);
        });
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

    #[test]
    fn owner_interrupt_is_owner_facing_for_visible_replies() {
        let task = crate::runtime_db::TaskRecord {
            id: 501,
            created_at: now_iso(),
            updated_at: now_iso(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: Some(17),
            source_channel: "attach_terminal".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: "Why no mail?".to_string(),
            detail: "Reply visibly to the owner.".to_string(),
            trust_level: "system".to_string(),
            priority_score: 1000,
            status: "blocked".to_string(),
            run_count: 1,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: Some("Der Mailentwurf liegt bereit.".to_string()),
        };

        assert!(is_owner_facing_task(&task));
    }

    #[test]
    fn owner_visible_boot_reply_filters_known_loop_noise() {
        assert!(owner_visible_boot_reply_text("Ja, ich bin dran.").is_some());
        assert!(owner_visible_boot_reply_text(
            "All systems nominal. Awaiting next command."
        )
        .is_none());
        assert!(owner_visible_boot_reply_text(
            "Workspace execution contract is active and no exec/browser machine path ran in this turn."
        )
        .is_none());
        assert!(owner_visible_boot_reply_text(
            "I am currently unable to produce any new progress for task #963."
        )
        .is_none());
    }

    #[test]
    fn simple_owner_chat_can_complete_directly_without_review() {
        with_temp_runtime("owner-chat-direct-complete", |paths| {
            let task = crate::runtime_db::TaskRecord {
                id: 777,
                created_at: now_iso(),
                updated_at: now_iso(),
                parent_task_id: None,
                worker_job_id: None,
                source_interrupt_id: Some(21),
                source_channel: "attach_terminal".to_string(),
                speaker: "Michael Welsch".to_string(),
                task_kind: "owner_interrupt".to_string(),
                title: "Chatten".to_string(),
                detail: "An was arbeitest du gerade?".to_string(),
                trust_level: "owner".to_string(),
                priority_score: 1000,
                status: "active".to_string(),
                run_count: 0,
                last_checkpoint_summary: None,
                last_checkpoint_at: None,
                last_output: None,
            };

            assert!(owner_interrupt_can_complete_directly(
                paths,
                &task,
                "continue",
                "reprioritize",
                false,
                "Ich arbeite gerade an der Chat- und Interrupt-Stabilisierung."
            ));
        });
    }

    #[test]
    fn substantive_owner_workspace_task_does_not_complete_directly() {
        with_temp_runtime("owner-workspace-no-direct-complete", |paths| {
            let task = crate::runtime_db::TaskRecord {
                id: 778,
                created_at: now_iso(),
                updated_at: now_iso(),
                parent_task_id: None,
                worker_job_id: None,
                source_interrupt_id: Some(22),
                source_channel: "attach_terminal".to_string(),
                speaker: "Michael Welsch".to_string(),
                task_kind: "owner_interrupt".to_string(),
                title: "Chatten".to_string(),
                detail: "Erstelle eine C++-Konsolenanwendung mit persistenter Speicherung.".to_string(),
                trust_level: "owner".to_string(),
                priority_score: 1000,
                status: "active".to_string(),
                run_count: 0,
                last_checkpoint_summary: None,
                last_checkpoint_at: None,
                last_output: None,
            };
            crate::runtime_db::record_context_package(
                paths,
                task.id,
                &task.title,
                "working",
                65536,
                "test",
                r#"{"rawInclusions":[{"sourceKind":"system_capability_contract","sourceRef":"contracts/system/workspace-execution-capability-policy.json","content":"workspace policy"}]}"#,
            )
            .expect("context package should persist");

            assert!(!owner_interrupt_can_complete_directly(
                paths,
                &task,
                "continue",
                "reprioritize",
                false,
                "Ich habe die Aufgabe aufgenommen."
            ));
        });
    }

    #[test]
    fn substantive_task_uses_direct_distillation_without_context_preparation() {
        with_temp_runtime("direct-distillation-routing", |paths| {
            let parent = crate::runtime_db::enqueue_internal_task(
                paths,
                None,
                "workspace_repair",
                "Repair the workspace",
                "Implement the concrete coding task with verified progress.",
                900,
            )
            .expect("should enqueue parent task");

            let fresh = classify_context_preparation_need(paths, &parent)
                .expect("classification should succeed");
            assert!(matches!(fresh, ContextPreparationRouting::Proceed));
        });
    }

    #[test]
    fn context_preparation_never_enqueues_historical_research_followup() {
        assert!(!should_enqueue_history_research(
            "context_preparation",
            true
        ));
        assert!(!should_enqueue_history_research(
            "historical_research",
            true
        ));
        assert!(!should_enqueue_history_research("self_review", true));
        assert!(!should_enqueue_history_research("worker_review", true));
        assert!(should_enqueue_history_research("owner_interrupt", true));
        assert!(!should_enqueue_history_research("owner_interrupt", false));
    }

    #[test]
    fn substantive_owner_work_keeps_historical_reload_inline() {
        with_temp_runtime("inline-history-on-owner-work", |paths| {
            let task = crate::runtime_db::TaskRecord {
                id: 37,
                created_at: now_iso(),
                updated_at: now_iso(),
                parent_task_id: None,
                worker_job_id: None,
                source_interrupt_id: Some(11),
                source_channel: "attach_terminal".to_string(),
                speaker: "Michael Welsch".to_string(),
                task_kind: "owner_interrupt".to_string(),
                title: "Stay on the C++ mission".to_string(),
                detail: "Keep the same substantive task active.".to_string(),
                trust_level: "owner".to_string(),
                priority_score: 1000,
                status: "active".to_string(),
                run_count: 7,
                last_checkpoint_summary: None,
                last_checkpoint_at: None,
                last_output: None,
            };

            assert!(should_keep_history_research_inline(
                paths,
                &task,
                true,
                "Need an older checkpoint before the next bounded step.",
            ));
        });
    }

    #[test]
    fn prepared_context_artifact_ready_requires_required_blocks() {
        with_temp_runtime("context-preparation-ready", |paths| {
            let incomplete = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step: "Open the repo root and inspect the first build file."
                    .to_string(),
                questions: Vec::new(),
                blocks: vec![crate::context_controller::ContextPreparedBlock {
                    block_id: "goal_and_authority".to_string(),
                    title: "Goal And Authority".to_string(),
                    token_budget: 180,
                    content: "The owner wants the C++ console app built.".to_string(),
                    why_included: "It is the current objective.".to_string(),
                    approx_tokens: None,
                    evidence_refs: vec!["task:22".to_string()],
                    omitted_items: Vec::new(),
                }],
                review: crate::context_controller::ContextPreparedReview {
                    decision: "go".to_string(),
                    note: "Looks fine.".to_string(),
                    missing_evidence: Vec::new(),
                    weak_blocks: Vec::new(),
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };
            assert!(!validate_prepared_context_artifact(&incomplete, &[], paths).ready);

            let complete = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step:
                    "Open the repo root and inspect the first build file before implementation."
                        .to_string(),
                questions: Vec::new(),
                blocks: vec![
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "goal_and_authority".to_string(),
                        title: "Goal And Authority".to_string(),
                        token_budget: 180,
                        content: "The owner requested the C++ console app implementation."
                            .to_string(),
                        why_included: "This is the task objective.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:22".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "definition_of_done".to_string(),
                        title: "Definition Of Done".to_string(),
                        token_budget: 220,
                        content:
                            "Done means registration, login, friend requests, direct messages, persistent history, and thread-safe processing all work together."
                                .to_string(),
                        why_included: "This is the completion contract.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:22".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "verified_world_state".to_string(),
                        title: "Verified World State".to_string(),
                        token_budget: 320,
                        content:
                            "The current task is an owner interrupt for the C++ app and the local workspace still needs initial repo inspection."
                                .to_string(),
                        why_included: "It grounds the first execution step.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:22".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "next_action_only".to_string(),
                        title: "Next Action Only".to_string(),
                        token_budget: 170,
                        content:
                            "List the workspace files, find the build system, and identify whether a C++ scaffold already exists."
                                .to_string(),
                        why_included: "It is the immediate next bounded step.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:22".to_string()],
                        omitted_items: Vec::new(),
                    },
                ],
                review: crate::context_controller::ContextPreparedReview {
                    decision: "go".to_string(),
                    note: "Required blocks are present and no blocking evidence gap remains."
                        .to_string(),
                    missing_evidence: Vec::new(),
                    weak_blocks: Vec::new(),
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };
            let rewrite_draft = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step:
                    "Open the repo root and inspect the first build file before implementation."
                        .to_string(),
                questions: vec![crate::context_controller::ContextQueryQuestion {
                    question_id: "repo_root".to_string(),
                    question: "Which verified workspace root contains the C++ app task?"
                        .to_string(),
                    why: "The rewrite draft must stay grounded in the real workspace.".to_string(),
                    query_mode: "sqlite_hybrid".to_string(),
                    source_kinds: vec!["task_detail".to_string()],
                    max_matches: Some(2),
                    required_keywords: vec!["c++".to_string(), "workspace".to_string()],
                }],
                blocks: complete.blocks.clone(),
                review: crate::context_controller::ContextPreparedReview {
                    decision: "revise".to_string(),
                    note: "The draft blocks exist, but the handoff still needs review.".to_string(),
                    missing_evidence: Vec::new(),
                    weak_blocks: Vec::new(),
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };
            let checkpoints = vec![crate::runtime_db::TaskCheckpointRecord {
                task_id: 22,
                created_at: "2026-03-20T00:00:00Z".to_string(),
                checkpoint_kind: "context_rewrite_draft".to_string(),
                summary: "Drafted the rewritten context blocks.".to_string(),
                detail: serde_json::json!({ "preparedContextArtifact": rewrite_draft }).to_string(),
            }];
            assert!(validate_prepared_context_artifact(&complete, &checkpoints, paths).ready);
        });
    }

    #[test]
    fn prepared_context_validation_requires_provenance_and_budget() {
        with_temp_runtime("context-preparation-budget", |paths| {
            let invalid = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step: "Inspect the repo root.".to_string(),
                questions: Vec::new(),
                blocks: vec![
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "goal_and_authority".to_string(),
                        title: "Goal And Authority".to_string(),
                        token_budget: 180,
                        content: "x".repeat(900),
                        why_included: "Needed.".to_string(),
                        approx_tokens: Some(400),
                        evidence_refs: Vec::new(),
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "definition_of_done".to_string(),
                        title: "Definition Of Done".to_string(),
                        token_budget: 220,
                        content: "Real completion criteria.".to_string(),
                        why_included: "Needed.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:22".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "verified_world_state".to_string(),
                        title: "Verified World State".to_string(),
                        token_budget: 320,
                        content: "Verified task evidence.".to_string(),
                        why_included: "Needed.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:22".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "next_action_only".to_string(),
                        title: "Next Action Only".to_string(),
                        token_budget: 170,
                        content: "Inspect the repo root.".to_string(),
                        why_included: "Needed.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:22".to_string()],
                        omitted_items: Vec::new(),
                    },
                ],
                review: crate::context_controller::ContextPreparedReview {
                    decision: "go".to_string(),
                    note: "Looks good.".to_string(),
                    missing_evidence: Vec::new(),
                    weak_blocks: Vec::new(),
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };
            let validation = validate_prepared_context_artifact(&invalid, &[], paths);
            assert!(!validation.ready);
            assert!(
                validation
                    .issues
                    .iter()
                    .any(|issue| issue.contains("no `evidenceRefs`"))
            );
            assert!(
                validation
                    .issues
                    .iter()
                    .any(|issue| issue.contains("exceeds token budget"))
            );
        });
    }

    #[test]
    fn prepared_context_validation_requires_weak_block_progress() {
        with_temp_runtime("context-preparation-weak-blocks", |paths| {
            let prior = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step: "Verify the repo root.".to_string(),
                questions: Vec::new(),
                blocks: vec![
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "goal_and_authority".to_string(),
                        title: "Goal And Authority".to_string(),
                        token_budget: 180,
                        content: "The owner wants the browser repair stabilized.".to_string(),
                        why_included: "Objective.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:55".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "definition_of_done".to_string(),
                        title: "Definition Of Done".to_string(),
                        token_budget: 220,
                        content: "Done means the browser capability can be executed safely."
                            .to_string(),
                        why_included: "Completion contract.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:55".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "verified_world_state".to_string(),
                        title: "Verified World State".to_string(),
                        token_budget: 320,
                        content: "The desktop-session bridge still needs proof.".to_string(),
                        why_included: "Grounding.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:55".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "next_action_only".to_string(),
                        title: "Next Action Only".to_string(),
                        token_budget: 170,
                        content: "Inspect the browser bridge command path.".to_string(),
                        why_included: "Next step.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:55".to_string()],
                        omitted_items: Vec::new(),
                    },
                ],
                review: crate::context_controller::ContextPreparedReview {
                    decision: "revise".to_string(),
                    note: "The verified world state block is still weak.".to_string(),
                    missing_evidence: Vec::new(),
                    weak_blocks: vec!["verified_world_state".to_string()],
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };
            let checkpoints = vec![crate::runtime_db::TaskCheckpointRecord {
                task_id: 55,
                created_at: "2026-03-20T00:00:00Z".to_string(),
                checkpoint_kind: "context_rewrite_review".to_string(),
                summary: "Review requested a better verified world state block.".to_string(),
                detail: serde_json::json!({ "preparedContextArtifact": prior }).to_string(),
            }];
            let unchanged = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step: "Verify the repo root.".to_string(),
                questions: Vec::new(),
                blocks: vec![
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "goal_and_authority".to_string(),
                        title: "Goal And Authority".to_string(),
                        token_budget: 180,
                        content: "The owner wants the browser repair stabilized.".to_string(),
                        why_included: "Objective.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:55".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "definition_of_done".to_string(),
                        title: "Definition Of Done".to_string(),
                        token_budget: 220,
                        content: "Done means the browser capability can be executed safely."
                            .to_string(),
                        why_included: "Completion contract.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:55".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "verified_world_state".to_string(),
                        title: "Verified World State".to_string(),
                        token_budget: 320,
                        content: "The desktop-session bridge still needs proof.".to_string(),
                        why_included: "Grounding.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:55".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "next_action_only".to_string(),
                        title: "Next Action Only".to_string(),
                        token_budget: 170,
                        content: "Inspect the browser bridge command path.".to_string(),
                        why_included: "Next step.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:55".to_string()],
                        omitted_items: Vec::new(),
                    },
                ],
                review: crate::context_controller::ContextPreparedReview {
                    decision: "revise".to_string(),
                    note: "Still revising.".to_string(),
                    missing_evidence: Vec::new(),
                    weak_blocks: vec!["verified_world_state".to_string()],
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };
            let validation = validate_prepared_context_artifact(&unchanged, &checkpoints, paths);
            assert!(!validation.ready);
            assert!(
                validation
                    .issues
                    .iter()
                    .any(|issue| issue.contains("weak-block revision"))
            );
        });
    }

    #[test]
    fn query_plan_phase_must_not_emit_blocks() {
        with_temp_runtime("context-preparation-query-phase-shape", |paths| {
            let invalid = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step: "Ask the context store for the strongest repo root evidence."
                    .to_string(),
                questions: vec![crate::context_controller::ContextQueryQuestion {
                    question_id: "repo_root".to_string(),
                    question: "Which verified repo root contains the C++ app?".to_string(),
                    why: "The next phase needs a grounded workspace.".to_string(),
                    query_mode: "sqlite_hybrid".to_string(),
                    source_kinds: vec!["task_detail".to_string()],
                    max_matches: Some(2),
                    required_keywords: vec!["c++".to_string(), "repo".to_string()],
                }],
                blocks: vec![crate::context_controller::ContextPreparedBlock {
                    block_id: "goal_and_authority".to_string(),
                    title: "Goal And Authority".to_string(),
                    token_budget: 180,
                    content: "The owner wants the C++ console app built.".to_string(),
                    why_included: "Objective.".to_string(),
                    approx_tokens: None,
                    evidence_refs: vec!["task:25".to_string()],
                    omitted_items: Vec::new(),
                }],
                review: crate::context_controller::ContextPreparedReview {
                    decision: "query_more".to_string(),
                    note: "Still gathering evidence.".to_string(),
                    missing_evidence: vec!["Verified repo root".to_string()],
                    weak_blocks: Vec::new(),
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };

            let validation = validate_prepared_context_artifact(&invalid, &[], paths);
            assert!(!validation.ready);
            assert!(
                validation
                    .issues
                    .iter()
                    .any(|issue| issue.contains("query_plan phase must not emit rewritten blocks"))
            );
        });
    }

    #[test]
    fn query_plan_can_report_missing_evidence_without_failing_contract() {
        with_temp_runtime("context-preparation-query-missing-evidence", |paths| {
            let artifact = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step: "Ask the context store for the verified repo root."
                    .to_string(),
                questions: vec![crate::context_controller::ContextQueryQuestion {
                    question_id: "repo_root".to_string(),
                    question: "Which verified repo root contains the C++ app?".to_string(),
                    why: "The rewrite phase needs a grounded workspace.".to_string(),
                    query_mode: "sqlite_hybrid".to_string(),
                    source_kinds: vec!["task_detail".to_string()],
                    max_matches: Some(2),
                    required_keywords: vec!["c++".to_string(), "repo".to_string()],
                }],
                blocks: Vec::new(),
                review: crate::context_controller::ContextPreparedReview {
                    decision: "query_more".to_string(),
                    note: "Need the grounded repo root before rewriting blocks.".to_string(),
                    missing_evidence: vec!["Verified repo root".to_string()],
                    weak_blocks: Vec::new(),
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };

            let validation = validate_prepared_context_artifact(&artifact, &[], paths);
            assert!(!validation.ready);
            assert!(
                !validation
                    .issues
                    .iter()
                    .any(|issue| issue.contains("missing required block"))
            );
            assert!(
                !validation
                    .issues
                    .iter()
                    .any(|issue| issue.contains("review still reports missing evidence"))
            );
        });
    }

    #[test]
    fn rewrite_phase_must_pass_through_review_before_go() {
        with_temp_runtime("context-preparation-rewrite-phase-go", |paths| {
            let prior = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step: "Ask the context store for the verified repo root."
                    .to_string(),
                questions: vec![crate::context_controller::ContextQueryQuestion {
                    question_id: "repo_root".to_string(),
                    question: "Which verified repo root contains the C++ app?".to_string(),
                    why: "The rewrite phase needs a grounded workspace.".to_string(),
                    query_mode: "sqlite_hybrid".to_string(),
                    source_kinds: vec!["task_detail".to_string()],
                    max_matches: Some(2),
                    required_keywords: vec!["c++".to_string(), "repo".to_string()],
                }],
                blocks: Vec::new(),
                review: crate::context_controller::ContextPreparedReview {
                    decision: "query_more".to_string(),
                    note: "Need retrieval evidence first.".to_string(),
                    missing_evidence: vec!["Verified repo root".to_string()],
                    weak_blocks: vec!["verified_world_state".to_string()],
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };
            let checkpoints = vec![crate::runtime_db::TaskCheckpointRecord {
                task_id: 25,
                created_at: "2026-03-20T00:00:00Z".to_string(),
                checkpoint_kind: "context_query_refine".to_string(),
                summary: "Prepared the retrieval questions.".to_string(),
                detail: serde_json::json!({ "preparedContextArtifact": prior }).to_string(),
            }];
            let invalid = crate::context_controller::ContextPreparedArtifact {
                immediate_next_step:
                    "Inspect the repo root and choose the first module boundary.".to_string(),
                questions: vec![crate::context_controller::ContextQueryQuestion {
                    question_id: "repo_root".to_string(),
                    question: "Which verified repo root contains the C++ app?".to_string(),
                    why: "The rewrite phase needs a grounded workspace.".to_string(),
                    query_mode: "sqlite_hybrid".to_string(),
                    source_kinds: vec!["task_detail".to_string()],
                    max_matches: Some(2),
                    required_keywords: vec!["c++".to_string(), "repo".to_string()],
                }],
                blocks: vec![
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "goal_and_authority".to_string(),
                        title: "Goal And Authority".to_string(),
                        token_budget: 180,
                        content: "The owner wants the C++ console app implemented.".to_string(),
                        why_included: "Objective.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:25".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "definition_of_done".to_string(),
                        title: "Definition Of Done".to_string(),
                        token_budget: 220,
                        content: "Done means auth, friends, messages, persistence, and thread safety all work together.".to_string(),
                        why_included: "Completion contract.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:25".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "verified_world_state".to_string(),
                        title: "Verified World State".to_string(),
                        token_budget: 320,
                        content: "The current task is a fresh owner interrupt for the C++ app.".to_string(),
                        why_included: "Grounding.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:25".to_string()],
                        omitted_items: Vec::new(),
                    },
                    crate::context_controller::ContextPreparedBlock {
                        block_id: "next_action_only".to_string(),
                        title: "Next Action Only".to_string(),
                        token_budget: 170,
                        content: "Inspect the workspace files and build system before writing code."
                            .to_string(),
                        why_included: "Immediate next step.".to_string(),
                        approx_tokens: None,
                        evidence_refs: vec!["task:25".to_string()],
                        omitted_items: Vec::new(),
                    },
                ],
                review: crate::context_controller::ContextPreparedReview {
                    decision: "go".to_string(),
                    note: "Looks ready.".to_string(),
                    missing_evidence: Vec::new(),
                    weak_blocks: Vec::new(),
                    budget_violations: Vec::new(),
                    repeated_from_prior: false,
                    findings: Vec::new(),
                    assessment: None,
                },
            };

            let validation = validate_prepared_context_artifact(&invalid, &checkpoints, paths);
            assert!(!validation.ready);
            assert!(
                validation
                    .issues
                    .iter()
                    .any(|issue| issue.contains("rewrite phase must hand off to review"))
            );
        });
    }

    #[test]
    fn context_preparation_checkpoint_kind_follows_active_phase() {
        assert_eq!(
            context_preparation_checkpoint_kind_for_phase("query_plan"),
            "context_query_plan"
        );
        assert_eq!(
            context_preparation_checkpoint_kind_for_phase("rewrite"),
            "context_rewrite_draft"
        );
        assert_eq!(
            context_preparation_checkpoint_kind_for_phase("review"),
            "context_rewrite_review"
        );
    }

    #[test]
    fn prepared_context_artifact_is_appended_to_checkpoint_detail() {
        let artifact = crate::context_controller::ContextPreparedArtifact {
            immediate_next_step: "Inspect the verified workspace root.".to_string(),
            questions: vec![crate::context_controller::ContextQueryQuestion {
                question_id: "workspace_root".to_string(),
                question: "Which repo root should the C++ app use?".to_string(),
                why: "Execution needs the right workspace.".to_string(),
                query_mode: "sqlite_hybrid".to_string(),
                source_kinds: vec!["task_detail".to_string()],
                max_matches: Some(2),
                required_keywords: vec!["c++".to_string(), "workspace".to_string()],
            }],
            blocks: Vec::new(),
            review: crate::context_controller::ContextPreparedReview {
                decision: "query_more".to_string(),
                note: "Need the grounded workspace root first.".to_string(),
                missing_evidence: vec!["Verified workspace root".to_string()],
                weak_blocks: vec!["verified_world_state".to_string()],
                budget_violations: Vec::new(),
                repeated_from_prior: false,
                findings: Vec::new(),
                assessment: None,
            },
        };
        let mut detail = "Narrative summary about the missing repository evidence.".to_string();
        append_prepared_context_artifact_to_detail(&mut detail, Some(&artifact));
        let extracted = extract_prepared_context_artifact_from_detail(&detail)
            .expect("artifact should round-trip");
        assert_eq!(extracted.immediate_next_step, artifact.immediate_next_step);
        assert_eq!(extracted.questions.len(), 1);
        assert_eq!(extracted.review.decision, "query_more");
    }
}

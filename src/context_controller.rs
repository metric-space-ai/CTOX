use crate::contracts::BootEntry;
use crate::contracts::ContextModePolicy;
use crate::contracts::ExecutionAuthorityPolicy;
use crate::contracts::LoopSafetyPolicy;
use crate::contracts::ModeSystemPolicy;
use crate::contracts::Paths;
use crate::contracts::describe_browser_vision_kleinhirn_selection;
use crate::contracts::describe_local_kleinhirn_candidates;
use crate::contracts::describe_kleinhirn_selection;
use crate::contracts::load_boot_entries;
use crate::contracts::load_census;
use crate::contracts::load_context_governance_policy;
use crate::contracts::load_context_policy;
use crate::contracts::load_execution_authority_policy;
use crate::contracts::load_installation_bootstrap_state;
use crate::contracts::load_loop_safety_policy;
use crate::contracts::load_model_policy;
use crate::contracts::load_mode_system_policy;
use crate::contracts::load_self_preservation_state;
use crate::contracts::now_iso;
use crate::command_exec::snapshot_sessions;
use crate::brain_runtime::load_kleinhirn_runtime_snapshot;
use crate::brain_runtime::grosshirn_runtime_configured;
use crate::brain_runtime::inspect_runtime_disk_headroom;
use crate::brain_runtime::local_browser_vision_kleinhirn_upgrade_available;
use crate::brain_runtime::local_kleinhirn_upgrade_available;
use crate::runtime_db::BiosDialogueEntry;
use crate::runtime_db::LearningEntryRecord;
use crate::runtime_db::MemoryItemRecord;
use crate::runtime_db::PersonProfileRecord;
use crate::runtime_db::TaskCheckpointRecord;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::TurnSignalRecord;
use crate::runtime_db::load_brain_routing_state;
use crate::runtime_db::load_external_brain_cost_summary;
use crate::runtime_db::list_active_learning_entries;
use crate::runtime_db::list_bios_dialogue;
use crate::runtime_db::list_memory_items;
use crate::runtime_db::list_pending_proactive_contact_candidates;
use crate::runtime_db::list_person_notes_for_person;
use crate::runtime_db::list_person_profiles;
use crate::runtime_db::list_recent_mail_previews_for_person;
use crate::runtime_db::list_skills;
use crate::runtime_db::list_task_checkpoints;
use crate::runtime_db::list_turn_signals_for_task;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::load_memory_summary;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::mark_learning_entries_recalled;
use crate::runtime_db::record_context_package;
use crate::runtime_db::sync_skills;
use serde::Serialize;
use std::collections::HashSet;
use std::fs;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPackage {
    pub created_at: String,
    pub task_id: i64,
    pub task_title: String,
    pub task_kind: String,
    pub source_channel: String,
    pub speaker: String,
    pub context_mode: String,
    pub budget_hint: usize,
    pub rationale: String,
    pub current_agent_mode: String,
    pub allowed_next_modes: Vec<String>,
    pub preferred_operating_goal: String,
    pub loop_safety: ContextLoopSafetySummary,
    pub execution_authority: ContextExecutionAuthoritySummary,
    pub brain_access: ContextBrainAccessSummary,
    pub skill_system: ContextSkillSystemSummary,
    pub context_governance: ContextGovernanceSummary,
    pub self_preservation_stage: ContextSelfPreservationStage,
    pub host_survival: ContextHostSurvivalSummary,
    pub focus_state: ContextFocusSnapshot,
    pub owner_calibration_summary: Option<String>,
    pub learning_summaries: ContextLearningSummaryBlock,
    pub people_working_set: Option<String>,
    pub task_brief: ContextTaskBrief,
    pub recent_boot_entries: Vec<ContextBootEntry>,
    pub recent_bios_dialogue: Vec<ContextDialogueEntry>,
    pub relevant_memory_items: Vec<ContextMemoryEntry>,
    pub relevant_learning_entries: Vec<ContextLearningEntry>,
    pub relevant_people: Vec<ContextPersonEntry>,
    pub pending_proactive_contacts: Vec<ContextProactiveContactEntry>,
    pub recent_task_checkpoints: Vec<ContextCheckpointEntry>,
    pub recent_turn_signals: Vec<ContextTurnSignalEntry>,
    pub exec_sessions: Vec<ContextExecSessionEntry>,
    pub available_skills: Vec<ContextSkillEntry>,
    pub raw_inclusions: Vec<ContextRawInclusion>,
    pub retrieval_notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextFocusSnapshot {
    pub mode: String,
    pub active_task_id: Option<i64>,
    pub active_task_title: String,
    pub queue_depth: i64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextTaskBrief {
    pub title: String,
    pub detail: String,
    pub trust_level: String,
    pub priority_score: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextLoopSafetySummary {
    pub principle: String,
    pub priority_law: String,
    pub owner_override_priority_floor: i64,
    pub self_preservation_priority_floor: i64,
    pub request_resources_after_run_count: i64,
    pub hard_block_after_run_count: i64,
    pub continue_same_task_requires_progress: bool,
    pub delegate_when_capable: bool,
    pub hard_block_when_unproductive: bool,
    pub known_failure_modes: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextExecutionAuthoritySummary {
    pub principle: String,
    pub root_agent_runtime_profile: String,
    pub root_agent_sandbox_enabled: bool,
    pub root_agent_full_machine_access: bool,
    pub delegated_worker_runtime_profile: String,
    pub delegated_workers_sandboxed: bool,
    pub delegated_workers_require_cto_approval: bool,
    pub escalation_rule: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextBrainAccessSummary {
    pub brain_access_mode: String,
    pub active_brain_route: String,
    pub current_runtime_kleinhirn: String,
    pub local_selected_kleinhirn: String,
    pub local_upgrade_available: bool,
    pub browser_vision_selected_kleinhirn: String,
    pub browser_vision_upgrade_available: bool,
    pub local_kleinhirn_candidates: Vec<String>,
    pub grosshirn_candidates: Vec<String>,
    pub grosshirn_configured: bool,
    pub grosshirn_boost_task_id: Option<i64>,
    pub grosshirn_boost_task_title: Option<String>,
    pub grosshirn_boost_expires_at: Option<String>,
    pub grosshirn_cooldown_until: Option<String>,
    pub external_grosshirn_calls: i64,
    pub external_grosshirn_total_tokens: i64,
    pub external_grosshirn_estimated_cost_usd: f64,
    pub external_grosshirn_last_call_at: Option<String>,
    pub operating_rule: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextSkillSystemSummary {
    pub repo_skill_root: String,
    pub sync_rule: String,
    pub operating_rule: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextSkillEntry {
    pub name: String,
    pub path: String,
    pub status: String,
    pub description: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextGovernanceSummary {
    pub principle: String,
    pub normal_context_control: String,
    pub historical_research_allowed: bool,
    pub agent_may_question_bad_compaction: bool,
    pub emergency_boundary: String,
    pub normal_actions: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextSelfPreservationStage {
    pub current_stage: String,
    pub guardrails_enabled: bool,
    pub agent_may_relax_bootstrap_guards: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextHostSurvivalSummary {
    pub disk_status: String,
    pub disk_mount_point: String,
    pub disk_available_gb: Option<u64>,
    pub disk_warning_floor_gb: u64,
    pub disk_critical_floor_gb: u64,
    pub operating_rule: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextBootEntry {
    pub timestamp: String,
    pub speaker: String,
    pub message: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextDialogueEntry {
    pub created_at: String,
    pub speaker: String,
    pub message: String,
    pub used_grosshirn: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextMemoryEntry {
    pub created_at: String,
    pub kind: String,
    pub summary: String,
    pub source: String,
    pub important: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextLearningSummaryBlock {
    pub working_set: Option<String>,
    pub operational: Option<String>,
    pub general: Option<String>,
    pub negative: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextLearningEntry {
    pub id: i64,
    pub learning_class: String,
    pub summary: String,
    pub applicability: String,
    pub confidence: f64,
    pub salience: i64,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPersonEntry {
    pub id: i64,
    pub display_name: String,
    pub primary_email: String,
    pub relationship_kind: String,
    pub trust_level: String,
    pub last_channel: String,
    pub interaction_count: i64,
    pub conversation_memory_summary: String,
    pub notebook_summary: String,
    pub proactive_guard_note: String,
    pub recent_notes: Vec<ContextPersonNoteEntry>,
    pub recent_mail_previews: Vec<ContextMailPreviewEntry>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPersonNoteEntry {
    pub note_kind: String,
    pub source_channel: String,
    pub summary: String,
    pub detail: String,
    pub important: bool,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextMailPreviewEntry {
    pub received_at: String,
    pub direction: String,
    pub subject: String,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextProactiveContactEntry {
    pub id: i64,
    pub person_display_name: String,
    pub person_email: String,
    pub status: String,
    pub channel: String,
    pub subject: String,
    pub rationale: String,
    pub conflict_check: String,
    pub validation_decision: String,
    pub validation_note: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCheckpointEntry {
    pub created_at: String,
    pub checkpoint_kind: String,
    pub summary: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextTurnSignalEntry {
    pub created_at: String,
    pub signal_kind: String,
    pub source_channel: String,
    pub speaker: String,
    pub message: String,
    pub turn_id: Option<i64>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextExecSessionEntry {
    pub session_id: String,
    pub status: String,
    pub cwd: String,
    pub tty: bool,
    pub command: Vec<String>,
    pub exit_code: Option<i32>,
    pub stdout: String,
    pub stderr: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextRawInclusion {
    pub source_kind: String,
    pub source_ref: String,
    pub content: String,
}

pub fn prepare_context_package(
    paths: &Paths,
    task: &TaskRecord,
) -> anyhow::Result<ContextPackage> {
    let policy = load_context_policy(paths);
    let mode_system = load_mode_system_policy(paths);
    let loop_safety = load_loop_safety_policy(paths);
    let execution_authority = load_execution_authority_policy(paths);
    let context_governance = load_context_governance_policy(paths);
    let self_preservation_state = load_self_preservation_state(paths);
    let model_policy = load_model_policy(paths);
    let census = load_census(paths);
    let current_runtime = load_kleinhirn_runtime_snapshot(paths);
    let owner_trust = load_owner_trust(paths).unwrap_or_default();
    let brain_routing = load_brain_routing_state(paths).unwrap_or_default();
    let external_brain_costs = load_external_brain_cost_summary(paths).unwrap_or_default();
    let mode = choose_mode(&policy, task);
    let focus = load_focus_state(paths)?;
    let allowed_next_modes = allowed_next_modes(&mode_system, &focus.mode);
    let available_skills = sync_skills(paths)
        .or_else(|_| list_skills(paths))
        .unwrap_or_default();
    let installation_bootstrap = load_installation_bootstrap_state(paths);
    let owner_summary = if mode.include_owner_summary {
        load_memory_summary(paths, "owner_calibration")?
    } else {
        None
    };
    let learning_summaries = ContextLearningSummaryBlock {
        working_set: load_memory_summary(paths, "learning_working_set")?,
        operational: load_memory_summary(paths, "learning_operational")?,
        general: load_memory_summary(paths, "learning_general")?,
        negative: load_memory_summary(paths, "learning_negative")?,
    };
    let people_working_set = load_memory_summary(paths, "people_working_set")?;
    let disk_headroom = inspect_runtime_disk_headroom(paths).ok();

    let keywords = extract_keywords(&format!("{} {}", task.title, task.detail));
    let boot_entries = select_boot_entries(load_boot_entries(paths), mode.recent_boot_entries);
    let bios_dialogue = select_dialogue_entries(
        list_bios_dialogue(paths, mode.recent_bios_dialogue.saturating_mul(2).max(1))?,
        &keywords,
        mode.recent_bios_dialogue,
    );
    let memory_items = select_memory_items(
        list_memory_items(paths, mode.recent_memory_items.saturating_mul(3).max(4))?,
        &keywords,
        mode.recent_memory_items,
    );
    let relevant_learning_entries = select_learning_entries(
        list_active_learning_entries(paths, mode.recent_memory_items.saturating_mul(4).max(8))?,
        &keywords,
        mode.recent_memory_items.saturating_add(2).max(4),
    );
    let relevant_people = select_relevant_people(
        list_person_profiles(paths, mode.recent_memory_items.saturating_mul(4).max(8))?,
        task,
        &keywords,
        3,
    );
    let pending_proactive_contacts =
        list_pending_proactive_contact_candidates(paths, 4).unwrap_or_default();
    let learning_ids = relevant_learning_entries
        .iter()
        .map(|entry| entry.id)
        .collect::<Vec<_>>();
    let _ = mark_learning_entries_recalled(paths, &learning_ids);
    let checkpoints = list_task_checkpoints(paths, task.id, mode.recent_task_checkpoints)?;
    let turn_signals =
        select_turn_signals(list_turn_signals_for_task(paths, task.id, 6)?, 3);
    let exec_sessions = select_exec_sessions(snapshot_sessions().unwrap_or_default(), task.id, 6);
    let mut raw_inclusions = build_raw_inclusions(task, &checkpoints, &bios_dialogue, &mode);
    if installation_bootstrap.status != "unconfigured" {
        let bootstrap_json =
            serde_json::to_string_pretty(&installation_bootstrap).unwrap_or_else(|_| "{}".to_string());
        raw_inclusions.push(ContextRawInclusion {
            source_kind: "installation_bootstrap".to_string(),
            source_ref: paths.installation_bootstrap_path.display().to_string(),
            content: trim_chars(&bootstrap_json, 1600),
        });
    }
    append_host_keyboard_inclusions(paths, task, &keywords, &mut raw_inclusions);
    let rationale = build_rationale(task, &mode.mode, keywords.len());

    let package = ContextPackage {
        created_at: now_iso(),
        task_id: task.id,
        task_title: task.title.clone(),
        task_kind: task.task_kind.clone(),
        source_channel: task.source_channel.clone(),
        speaker: task.speaker.clone(),
        context_mode: mode.mode.clone(),
        budget_hint: mode.budget_hint,
        rationale: rationale.clone(),
        current_agent_mode: focus.mode.clone(),
        allowed_next_modes,
        preferred_operating_goal: mode_system.preferred_operating_goal.clone(),
        loop_safety: build_loop_safety_summary(&loop_safety),
        execution_authority: build_execution_authority_summary(&execution_authority),
        brain_access: ContextBrainAccessSummary {
            brain_access_mode: owner_trust.brain_access_mode.clone(),
            active_brain_route: brain_routing.route_mode.clone(),
            current_runtime_kleinhirn: current_runtime
                .as_ref()
                .map(|snapshot| {
                    let label = snapshot
                        .official_label
                        .as_deref()
                        .or(snapshot.runtime_model.as_deref())
                        .unwrap_or("unknown");
                    let runtime = snapshot
                        .runtime_model
                        .as_deref()
                        .unwrap_or("unknown");
                    format!("{label} ({runtime})")
                })
                .unwrap_or_else(|| "unknown".to_string()),
            local_selected_kleinhirn: describe_kleinhirn_selection(&model_policy, &census),
            local_upgrade_available: local_kleinhirn_upgrade_available(paths),
            browser_vision_selected_kleinhirn: describe_browser_vision_kleinhirn_selection(
                &model_policy,
                &census,
            ),
            browser_vision_upgrade_available: local_browser_vision_kleinhirn_upgrade_available(
                paths,
            ),
            local_kleinhirn_candidates: describe_local_kleinhirn_candidates(
                &model_policy,
                &census,
            ),
            grosshirn_candidates: model_policy
                .grosshirn_candidates
                .iter()
                .map(|candidate| {
                    format!(
                        "{} ({})",
                        candidate.official_label, candidate.model_id
                    )
                })
                .collect(),
            grosshirn_configured: grosshirn_runtime_configured(paths),
            grosshirn_boost_task_id: brain_routing.boosted_task_id,
            grosshirn_boost_task_title: if brain_routing.boosted_task_title.trim().is_empty() {
                None
            } else {
                Some(brain_routing.boosted_task_title.clone())
            },
            grosshirn_boost_expires_at: brain_routing.boost_expires_at.clone(),
            grosshirn_cooldown_until: brain_routing.cooldown_until.clone(),
            external_grosshirn_calls: external_brain_costs.grosshirn_calls,
            external_grosshirn_total_tokens: external_brain_costs.total_tokens,
            external_grosshirn_estimated_cost_usd: external_brain_costs.estimated_cost_usd,
            external_grosshirn_last_call_at: external_brain_costs.last_external_call_at,
            operating_rule: "Kleinhirn ist der kostenlose lokale Default. Fuer screenshot- oder UI-wahrnehmungslastige Browserarbeit soll zuerst ein vision-faehiges lokales Qwen3.5-Kleinhirn erwogen werden. Externes Grosshirn ist nur ein temporaerer per-task Boost: du entscheidest selbst, wann die Kosten gerechtfertigt sind, haeltst lokalen Kleinhirn-Fallback bereit und gibst den Boost nach schwierigen Phasen wieder frei.".to_string(),
        },
        skill_system: ContextSkillSystemSummary {
            repo_skill_root: ".agents/skills".to_string(),
            sync_rule: "Repo-local skills are rescanned while preparing later turns, so any new SKILL.md written under .agents/skills becomes visible in the next context package.".to_string(),
            operating_rule: "When a reusable capability is missing, first inspect relevant repo-local skills. When you build a reusable tool or workflow, also create or update a repo-local operations skill so later turns can rediscover and operate it.".to_string(),
        },
        context_governance: ContextGovernanceSummary {
            principle: context_governance.principle,
            normal_context_control: "Normal context shaping is agent-controlled. The kernel only intervenes at hard physical overflow boundaries.".to_string(),
            historical_research_allowed: context_governance.historical_research_allowed,
            agent_may_question_bad_compaction: context_governance
                .agent_may_question_bad_compaction,
            emergency_boundary: "If the next LLM call would otherwise overflow, the kernel may temporarily shrink the package just enough to keep the loop alive.".to_string(),
            normal_actions: context_governance.normal_agent_actions,
        },
        self_preservation_stage: ContextSelfPreservationStage {
            current_stage: self_preservation_state.current_stage,
            guardrails_enabled: self_preservation_state.guardrails_enabled,
            agent_may_relax_bootstrap_guards: self_preservation_state
                .agent_may_relax_bootstrap_guards,
            notes: self_preservation_state.notes,
        },
        host_survival: ContextHostSurvivalSummary {
            disk_status: disk_headroom
                .as_ref()
                .map(|status| status.status.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            disk_mount_point: disk_headroom
                .as_ref()
                .map(|status| status.mount_point.clone())
                .unwrap_or_else(|| "unknown".to_string()),
            disk_available_gb: disk_headroom.as_ref().map(|status| status.available_gb),
            disk_warning_floor_gb: disk_headroom
                .as_ref()
                .map(|status| status.warning_floor_gb)
                .unwrap_or(12),
            disk_critical_floor_gb: disk_headroom
                .as_ref()
                .map(|status| status.critical_floor_gb)
                .unwrap_or(6),
            operating_rule: disk_headroom
                .as_ref()
                .map(|status| status.note())
                .unwrap_or_else(|| {
                    "If host disk headroom cannot be observed right now, treat storage survival as an open risk and verify it before heavy builds, model downloads or reinstalls.".to_string()
                }),
        },
        focus_state: ContextFocusSnapshot {
            mode: focus.mode,
            active_task_id: focus.active_task_id,
            active_task_title: focus.active_task_title,
            queue_depth: focus.queue_depth,
            note: focus.note,
        },
        owner_calibration_summary: owner_summary,
        learning_summaries,
        people_working_set,
        task_brief: ContextTaskBrief {
            title: task.title.clone(),
            detail: trim_chars(&task.detail, 700),
            trust_level: task.trust_level.clone(),
            priority_score: task.priority_score,
        },
        recent_boot_entries: boot_entries
            .into_iter()
            .map(|entry| ContextBootEntry {
                timestamp: entry.timestamp,
                speaker: entry.speaker,
                message: trim_chars(&entry.message, 320),
            })
            .collect(),
        recent_bios_dialogue: bios_dialogue
            .into_iter()
            .map(|entry| ContextDialogueEntry {
                created_at: entry.created_at,
                speaker: entry.speaker,
                message: trim_chars(&entry.message, 320),
                used_grosshirn: entry.used_grosshirn,
            })
            .collect(),
        relevant_memory_items: memory_items
            .into_iter()
            .map(|entry| ContextMemoryEntry {
                created_at: entry.created_at,
                kind: entry.kind,
                summary: trim_chars(&entry.summary, 220),
                source: entry.source,
                important: entry.important,
            })
            .collect(),
        relevant_learning_entries: relevant_learning_entries
            .into_iter()
            .map(|entry| ContextLearningEntry {
                id: entry.id,
                learning_class: entry.learning_class,
                summary: trim_chars(&entry.summary, 220),
                applicability: trim_chars(&entry.applicability, 220),
                confidence: entry.confidence,
                salience: entry.salience,
            })
            .collect(),
        relevant_people: relevant_people
            .into_iter()
            .map(|person| ContextPersonEntry {
                id: person.id,
                display_name: person.display_name.clone(),
                primary_email: person.primary_email.clone(),
                relationship_kind: person.relationship_kind.clone(),
                trust_level: person.trust_level.clone(),
                last_channel: person.last_channel.clone(),
                interaction_count: person.interaction_count,
                conversation_memory_summary: trim_chars(&person.conversation_memory_summary, 260),
                notebook_summary: trim_chars(&person.notebook_summary, 260),
                proactive_guard_note: trim_chars(&person.proactive_guard_note, 220),
                recent_notes: list_person_notes_for_person(paths, person.id, 3)
                    .unwrap_or_default()
                    .into_iter()
                    .map(|note| ContextPersonNoteEntry {
                        note_kind: note.note_kind,
                        source_channel: note.source_channel,
                        summary: trim_chars(&note.summary, 220),
                        detail: trim_chars(&note.detail, 320),
                        important: note.important,
                    })
                    .collect(),
                recent_mail_previews: list_recent_mail_previews_for_person(
                    paths,
                    &person.primary_email,
                    2,
                )
                .unwrap_or_default()
                .into_iter()
                .map(|mail| ContextMailPreviewEntry {
                    received_at: mail.received_at_iso,
                    direction: mail.direction,
                    subject: trim_chars(&mail.subject, 180),
                    preview: trim_chars(&mail.preview, 220),
                })
                .collect(),
            })
            .collect(),
        pending_proactive_contacts: pending_proactive_contacts
            .into_iter()
            .map(|candidate| ContextProactiveContactEntry {
                id: candidate.id,
                person_display_name: candidate.person_display_name,
                person_email: candidate.person_email,
                status: candidate.status,
                channel: candidate.channel,
                subject: trim_chars(&candidate.subject, 180),
                rationale: trim_chars(&candidate.rationale, 260),
                conflict_check: trim_chars(&candidate.conflict_check, 220),
                validation_decision: candidate.validation_decision,
                validation_note: trim_chars(&candidate.validation_note, 220),
            })
            .collect(),
        recent_task_checkpoints: checkpoints
            .iter()
            .map(|entry| ContextCheckpointEntry {
                created_at: entry.created_at.clone(),
                checkpoint_kind: entry.checkpoint_kind.clone(),
                summary: trim_chars(&entry.summary, 220),
                detail: trim_chars(&entry.detail, 420),
            })
            .collect(),
        recent_turn_signals: turn_signals
            .into_iter()
            .map(|signal| ContextTurnSignalEntry {
                created_at: signal.created_at,
                signal_kind: signal.signal_kind,
                source_channel: signal.source_channel,
                speaker: signal.speaker,
                message: trim_chars(&signal.message, 280),
                turn_id: signal.turn_id,
            })
            .collect(),
        exec_sessions,
        available_skills: available_skills
            .into_iter()
            .take(16)
            .map(|skill| ContextSkillEntry {
                name: skill.name,
                path: skill.path,
                status: skill.status,
                description: trim_chars(&skill.notes, 260),
            })
            .collect(),
        raw_inclusions,
        retrieval_notes: vec![
            "Nutze dieses Paket als aktiven Startkontext fuer genau einen bounded Task-Run.".to_string(),
            "Ziehe keine komplette Historie in den Kopf; rohe Vergangenheit soll nur gezielt nachgeladen werden.".to_string(),
            "Wenn eine alte Festlegung fehlt, fordere gezielt historische Nachladung an statt breit zu spekulieren.".to_string(),
            "Normale Kontextpflege, Kompaktierung und Nachladung sind deine agentischen Entscheidungen; der Kernel greift nur an harter Overflow-Grenze ein.".to_string(),
            "Exec-Sessions sind echte laufende Terminalkontexte. Wenn eine passende Session bereits existiert, kannst du sie gezielt per Session-ID weiterbenutzen statt neuen Shell-Zustand zu erfinden.".to_string(),
            "Wenn dir fuer ein lokales Kleinhirn-Upgrade GPU-, VRAM- oder mistralrs-tune-Evidenz fehlt, darfst du systemCensusAction=run setzen.".to_string(),
            "Wenn ein Installation-Bootstrap im Kontext liegt, behandle ihn als echte Owner-/Installer-Vorgabe fuer die fruehe Kommunikationsfaehigkeit und Kanalplanung.".to_string(),
            "Repo-Skills unter .agents/skills sind deine persistente Selbst-Erweiterungsflaeche. Wenn du dort einen neuen Skill schreibst, taucht er im naechsten Turn wieder im Skill-Katalog auf.".to_string(),
            "Wenn du ein neues Tool baust, schreibe zusaetzlich einen Operations-Skill mit den konkreten Kommandos, Pfaden und Fehlergrenzen fuer spaetere Turns.".to_string(),
            "hostSurvival gibt dir den aktuellen Disk-Headroom als CTO-Signal. Nutze ihn agentisch: wenn Speicher knapp wird, priorisiere bounded Inspektion, Kapazitaetsplanung und sichere Aufraeumarbeit selbst, statt blind weiter zu expandieren.".to_string(),
            "learningSummaries ist dein verdichtetes High-Level-Gedaechtnis. Nutze relevantLearningEntries, wenn du die Details eines frueheren Learnings wieder konkret anwenden musst.".to_string(),
            "peopleWorkingSet und relevantPeople halten hochverdichtete Personenpfade, Gespraechsnotizen und Mail-Referenzen bereit, damit du Menschen nicht wie stateless Tickets behandelst.".to_string(),
            "pendingProactiveContacts sind nur Entwuerfe oder Review-Faelle. Behaupte nie, dass ein proaktiver Vorschlag bereits versendet wurde, solange keine gesonderte Ausspielung stattgefunden hat.".to_string(),
            "Wenn du in diesem bounded Schritt ein neues belastbares Learning gewinnst, gib es als learningEntries aus statt es nur im Fliesstext zu verstecken.".to_string(),
        ],
    };

    let package_json = serde_json::to_string_pretty(&package)?;
    let _ = record_context_package(
        paths,
        task.id,
        &task.title,
        &package.context_mode,
        package.budget_hint,
        &rationale,
        &package_json,
    )?;
    Ok(package)
}

fn build_loop_safety_summary(policy: &LoopSafetyPolicy) -> ContextLoopSafetySummary {
    ContextLoopSafetySummary {
        principle: policy.principle.clone(),
        priority_law: policy.priority_law.clone(),
        owner_override_priority_floor: policy.owner_override_priority_floor,
        self_preservation_priority_floor: policy.self_preservation_priority_floor,
        request_resources_after_run_count: policy.request_resources_after_run_count,
        hard_block_after_run_count: policy.hard_block_after_run_count,
        continue_same_task_requires_progress: policy.continue_same_task_requires_progress,
        delegate_when_capable: policy.delegate_when_capable,
        hard_block_when_unproductive: policy.hard_block_when_unproductive,
        known_failure_modes: policy
            .failure_modes
            .iter()
            .map(|mode| format!("{}: {}", mode.key, mode.description))
            .collect(),
    }
}

fn build_execution_authority_summary(
    policy: &ExecutionAuthorityPolicy,
) -> ContextExecutionAuthoritySummary {
    ContextExecutionAuthoritySummary {
        principle: policy.principle.clone(),
        root_agent_runtime_profile: policy.root_agent.runtime_profile.clone(),
        root_agent_sandbox_enabled: policy.root_agent.sandbox_enabled,
        root_agent_full_machine_access: policy.root_agent.full_machine_access,
        delegated_worker_runtime_profile: policy.delegated_workers.runtime_profile.clone(),
        delegated_workers_sandboxed: policy.delegated_workers.sandbox_enabled,
        delegated_workers_require_cto_approval: policy
            .delegated_workers
            .approval_required_for_high_impact_actions,
        escalation_rule: policy.escalation_rule.clone(),
    }
}

fn select_turn_signals(entries: Vec<TurnSignalRecord>, limit: usize) -> Vec<TurnSignalRecord> {
    entries.into_iter().take(limit).collect()
}

fn select_exec_sessions(
    mut entries: Vec<crate::command_exec::SessionSnapshot>,
    task_id: i64,
    limit: usize,
) -> Vec<ContextExecSessionEntry> {
    let task_marker = format!("task-{task_id}");
    entries.sort_by(|left, right| {
        let left_active = left.status == "active";
        let right_active = right.status == "active";
        right_active
            .cmp(&left_active)
            .then_with(|| {
                let left_match = left.session_id.contains(&task_marker);
                let right_match = right.session_id.contains(&task_marker);
                right_match.cmp(&left_match)
            })
            .then_with(|| right.created_at.cmp(&left.created_at))
    });
    entries
        .into_iter()
        .take(limit)
        .map(|entry| ContextExecSessionEntry {
            session_id: entry.session_id,
            status: entry.status,
            cwd: entry.cwd,
            tty: entry.tty,
            command: entry.command,
            exit_code: entry.exit_code,
            stdout: trim_chars(&entry.stdout, 900),
            stderr: trim_chars(&entry.stderr, 900),
        })
        .collect()
}

fn choose_mode(
    policy: &crate::contracts::ContextPolicy,
    task: &TaskRecord,
) -> ContextModePolicy {
    let task_kind = task.task_kind.to_lowercase();
    let source_channel = task.source_channel.to_lowercase();
    let lowered_detail = task.detail.to_lowercase();
    let mode_name = if policy
        .system_channels
        .iter()
        .any(|channel| channel.eq_ignore_ascii_case(&source_channel))
        || policy
            .forensic_task_kinds
            .iter()
            .any(|kind| kind.eq_ignore_ascii_case(&task_kind))
        || lowered_detail.contains("superpassword")
        || lowered_detail.contains("branding")
        || lowered_detail.contains("bios")
    {
        "forensic"
    } else if policy
        .minimal_task_kinds
        .iter()
        .any(|kind| kind.eq_ignore_ascii_case(&task_kind))
    {
        "minimal"
    } else {
        policy.default_mode.as_str()
    };

    policy
        .modes
        .iter()
        .find(|mode| mode.mode == mode_name)
        .cloned()
        .unwrap_or_else(|| {
            policy
                .modes
                .iter()
                .find(|mode| mode.mode == policy.default_mode)
                .cloned()
                .unwrap_or(ContextModePolicy {
                    mode: "working".to_string(),
                    budget_hint: 1800,
                    recent_boot_entries: 2,
                    recent_bios_dialogue: 3,
                    recent_memory_items: 4,
                    recent_task_checkpoints: 2,
                    include_owner_summary: true,
                    include_raw_task_detail: true,
                })
        })
}

fn select_boot_entries(entries: Vec<BootEntry>, limit: usize) -> Vec<BootEntry> {
    if limit == 0 || entries.is_empty() {
        return Vec::new();
    }
    let len = entries.len();
    entries
        .into_iter()
        .skip(len.saturating_sub(limit))
        .collect()
}

fn select_dialogue_entries(
    entries: Vec<BiosDialogueEntry>,
    keywords: &HashSet<String>,
    limit: usize,
) -> Vec<BiosDialogueEntry> {
    select_by_keywords(entries, |entry| format!("{} {}", entry.speaker, entry.message), keywords, limit)
}

fn select_memory_items(
    entries: Vec<MemoryItemRecord>,
    keywords: &HashSet<String>,
    limit: usize,
) -> Vec<MemoryItemRecord> {
    select_by_keywords(entries, |entry| format!("{} {}", entry.summary, entry.detail), keywords, limit)
}

fn select_learning_entries(
    entries: Vec<LearningEntryRecord>,
    keywords: &HashSet<String>,
    limit: usize,
) -> Vec<LearningEntryRecord> {
    if limit == 0 || entries.is_empty() {
        return Vec::new();
    }
    let mut ranked: Vec<(usize, usize, LearningEntryRecord)> = entries
        .into_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let mut score = relevance_score(
                &format!("{} {} {}", entry.summary, entry.detail, entry.applicability),
                keywords,
            );
            score += match entry.learning_class.as_str() {
                "operational" => 9,
                "negative" => 6,
                _ => 3,
            };
            score += usize::try_from(entry.salience.max(0)).unwrap_or_default() / 10;
            score += (entry.confidence.clamp(0.0, 1.0) * 10.0).round() as usize;
            (score, idx, entry)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked
        .into_iter()
        .take(limit)
        .map(|(_, _, entry)| entry)
        .collect()
}

fn select_relevant_people(
    entries: Vec<PersonProfileRecord>,
    task: &TaskRecord,
    keywords: &HashSet<String>,
    limit: usize,
) -> Vec<PersonProfileRecord> {
    if limit == 0 || entries.is_empty() {
        return Vec::new();
    }
    let speaker_norm = normalize_person_identity(&task.speaker);
    let email_norm = if task.speaker.contains('@') {
        task.speaker.trim().to_lowercase()
    } else {
        String::new()
    };
    let mut ranked: Vec<(usize, usize, PersonProfileRecord)> = entries
        .into_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let haystack = format!(
                "{} {} {} {} {}",
                entry.display_name,
                entry.primary_email,
                entry.relationship_kind,
                entry.conversation_memory_summary,
                entry.notebook_summary,
            );
            let mut score = relevance_score(&haystack, keywords);
            if !speaker_norm.is_empty()
                && normalize_person_identity(&entry.display_name) == speaker_norm
            {
                score += 50;
            }
            if !email_norm.is_empty() && entry.primary_email.eq_ignore_ascii_case(&email_norm) {
                score += 50;
            }
            score += match entry.relationship_kind.as_str() {
                "owner" => 18,
                "reports_to" | "ceo" | "board" => 12,
                "peer_cxo" | "subordinate_person" => 8,
                _ => 3,
            };
            score += usize::try_from(entry.interaction_count.max(0)).unwrap_or_default().min(12);
            if task.task_kind == "person_relationship_review" {
                score += 8;
            }
            (score, idx, entry)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked
        .into_iter()
        .take(limit)
        .map(|(_, _, entry)| entry)
        .collect()
}

fn select_by_keywords<T, F>(
    entries: Vec<T>,
    to_text: F,
    keywords: &HashSet<String>,
    limit: usize,
) -> Vec<T>
where
    T: Clone,
    F: Fn(&T) -> String,
{
    if limit == 0 || entries.is_empty() {
        return Vec::new();
    }
    let mut ranked: Vec<(usize, usize, T)> = entries
        .into_iter()
        .enumerate()
        .map(|(idx, entry)| {
            let score = relevance_score(&to_text(&entry), keywords);
            (score, idx, entry)
        })
        .collect();
    ranked.sort_by(|a, b| b.0.cmp(&a.0).then(a.1.cmp(&b.1)));
    ranked.into_iter().take(limit).map(|(_, _, entry)| entry).collect()
}

fn normalize_person_identity(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn build_raw_inclusions(
    task: &TaskRecord,
    checkpoints: &[TaskCheckpointRecord],
    dialogue: &[BiosDialogueEntry],
    mode: &ContextModePolicy,
) -> Vec<ContextRawInclusion> {
    let mut raw = Vec::new();
    if mode.include_raw_task_detail {
        raw.push(ContextRawInclusion {
            source_kind: "task_detail".to_string(),
            source_ref: format!("task:{}", task.id),
            content: trim_chars(&task.detail, 900),
        });
    }
    let checkpoint_limit = if task.task_kind == "historical_research" {
        3
    } else {
        1
    };
    for checkpoint in checkpoints.iter().take(checkpoint_limit) {
        raw.push(ContextRawInclusion {
            source_kind: "task_checkpoint".to_string(),
            source_ref: format!("task:{}:{}", checkpoint.task_id, checkpoint.checkpoint_kind),
            content: trim_chars(&checkpoint.detail, 500),
        });
    }
    if mode.mode == "forensic" {
        let dialogue_limit = if task.task_kind == "historical_research" {
            3
        } else {
            1
        };
        for entry in dialogue.iter().take(dialogue_limit) {
            raw.push(ContextRawInclusion {
                source_kind: "bios_dialogue".to_string(),
                source_ref: format!("bios:{}", entry.created_at),
                content: trim_chars(&entry.message, 500),
            });
        }
    }
    raw
}

fn append_host_keyboard_inclusions(
    paths: &Paths,
    task: &TaskRecord,
    keywords: &HashSet<String>,
    raw_inclusions: &mut Vec<ContextRawInclusion>,
) {
    if !task_requires_host_keyboard_context(task, keywords) {
        return;
    }
    push_file_raw_inclusion(
        raw_inclusions,
        "repo_skill",
        paths.root.join(".agents/skills/host-keyboard-operations/SKILL.md"),
        1800,
    );
    push_file_raw_inclusion(
        raw_inclusions,
        "host_keyboard_contract",
        paths.system_dir.join("host-keyboard-capability-policy.json"),
        2200,
    );
}

fn push_file_raw_inclusion(
    raw_inclusions: &mut Vec<ContextRawInclusion>,
    source_kind: &str,
    path: std::path::PathBuf,
    limit: usize,
) {
    let Ok(content) = fs::read_to_string(&path) else {
        return;
    };
    raw_inclusions.push(ContextRawInclusion {
        source_kind: source_kind.to_string(),
        source_ref: path.display().to_string(),
        content: trim_chars(&content, limit),
    });
}

fn task_requires_host_keyboard_context(task: &TaskRecord, keywords: &HashSet<String>) -> bool {
    if task.task_kind != "owner_interrupt" {
        return false;
    }
    let text = format!("{} {}", task.title.to_lowercase(), task.detail.to_lowercase());
    let keyword_hit = keywords.iter().any(|keyword| {
        matches!(
            keyword.as_str(),
            "keyboard"
                | "layout"
                | "keymap"
                | "tastatur"
                | "deutsch"
                | "german"
                | "setxkbmap"
                | "localectl"
                | "loadkeys"
        )
    });
    keyword_hit
        || text.contains("keyboard")
        || text.contains("layout")
        || text.contains("keymap")
        || text.contains("tastatur")
        || text.contains("deutsch")
        || text.contains("german")
        || text.contains("setxkbmap")
        || text.contains("localectl")
        || text.contains("xkb")
        || text.contains("loadkeys")
}

fn build_rationale(task: &TaskRecord, mode: &str, keyword_count: usize) -> String {
    format!(
        "context_mode={} for task {} ({}) via {} with {} extracted topic keywords",
        mode, task.id, task.task_kind, task.source_channel, keyword_count
    )
}

fn allowed_next_modes(policy: &ModeSystemPolicy, current_mode: &str) -> Vec<String> {
    let modes: Vec<String> = policy
        .transitions
        .iter()
        .filter(|rule| rule.from_mode == current_mode)
        .map(|rule| rule.to_mode.clone())
        .collect();
    if modes.is_empty() {
        vec!["reprioritize".to_string(), "idle".to_string()]
    } else {
        modes
    }
}

fn extract_keywords(text: &str) -> HashSet<String> {
    text.split(|c: char| !c.is_alphanumeric())
        .map(|part| part.trim().to_lowercase())
        .filter(|part| part.len() >= 4)
        .take(16)
        .collect()
}

fn relevance_score(text: &str, keywords: &HashSet<String>) -> usize {
    let lowered = text.to_lowercase();
    let mut score = 0;
    for keyword in keywords {
        if lowered.contains(keyword) {
            score += 3;
        }
    }
    for token in [
        "owner",
        "bios",
        "homepage",
        "branding",
        "superpassword",
        "terminal",
        "kleinhirn",
        "grosshirn",
        "organigram",
    ] {
        if lowered.contains(token) {
            score += 2;
        }
    }
    score
}

fn trim_chars(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        value.to_string()
    } else {
        value.chars().take(limit).collect::<String>() + "..."
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime_db::TaskRecord;
    use std::path::PathBuf;

    fn sample_task(title: &str, detail: &str) -> TaskRecord {
        TaskRecord {
            id: 1,
            created_at: "2026-03-19T00:00:00Z".to_string(),
            updated_at: "2026-03-19T00:00:00Z".to_string(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: title.to_string(),
            detail: detail.to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "queued".to_string(),
            run_count: 0,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        }
    }

    fn temp_root(label: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "cto_context_keyboard_{}_{}_{}",
            label,
            std::process::id(),
            now_iso().replace(':', "-")
        ));
        std::fs::create_dir_all(root.join(".agents/skills/host-keyboard-operations")).unwrap();
        std::fs::create_dir_all(root.join("contracts/system")).unwrap();
        root
    }

    #[test]
    fn keyboard_owner_interrupt_gets_host_keyboard_context() {
        let task = sample_task(
            "Bitte Tastatur auf Deutsch umstellen",
            "Change the keyboard layout of this host to German and verify it.",
        );
        let keywords = extract_keywords(&format!("{} {}", task.title, task.detail));
        assert!(task_requires_host_keyboard_context(&task, &keywords));
    }

    #[test]
    fn host_keyboard_skill_and_contract_are_included_for_keyboard_task() {
        let root = temp_root("raw_inclusion");
        std::fs::write(
            root.join(".agents/skills/host-keyboard-operations/SKILL.md"),
            "# Host Keyboard Operations\n",
        )
        .unwrap();
        std::fs::write(
            root.join("contracts/system/host-keyboard-capability-policy.json"),
            "{\"version\":1,\"purpose\":\"keyboard\"}",
        )
        .unwrap();
        let paths = Paths {
            root: root.clone(),
            contracts_dir: root.join("contracts"),
            runtime_dir: root.join("runtime"),
            uploads_dir: root.join("runtime/uploads"),
            browser_artifacts_dir: root.join("runtime/browser"),
            recovery_dir: root.join("runtime/recovery"),
            history_dir: root.join("contracts/history"),
            models_dir: root.join("contracts/models"),
            homepage_dir: root.join("contracts/homepage"),
            bootstrap_dir: root.join("contracts/bootstrap"),
            context_dir: root.join("contracts/context"),
            system_dir: root.join("contracts/system"),
            browser_dir: root.join("contracts/browser"),
            genome_path: root.join("contracts/genome/genome.json"),
            bios_path: root.join("contracts/bios/bios.json"),
            org_path: root.join("contracts/org/organigram.json"),
            root_auth_path: root.join("contracts/root_auth/root_auth.json"),
            model_policy_path: root.join("contracts/models/model-policy.json"),
            homepage_policy_path: root.join("contracts/homepage/homepage-policy.json"),
            bootstrap_task_pack_path: root.join("contracts/bootstrap/bootstrap-task-pack.json"),
            installation_bootstrap_path: root.join("contracts/bootstrap/installation-bootstrap.json"),
            context_policy_path: root.join("contracts/context/context-policy.json"),
            context_governance_policy_path: root.join("contracts/context/context-governance-policy.json"),
            mode_system_policy_path: root.join("contracts/system/mode-system-policy.json"),
            loop_safety_policy_path: root.join("contracts/system/loop-safety-policy.json"),
            execution_authority_policy_path: root.join("contracts/system/execution-authority-policy.json"),
            browser_engine_policy_path: root.join("contracts/browser/browser-engine-policy.json"),
            browser_capability_policy_path: root.join("contracts/browser/browser-capability-policy.json"),
            browser_subworker_policy_path: root.join("contracts/browser/browser-subworker-policy.json"),
            self_preservation_state_path: root.join("contracts/system/self-preservation-state.json"),
            origin_story_path: root.join("contracts/history/origin-story.md"),
            creation_ledger_path: root.join("contracts/history/creation-ledger.md"),
            boot_log_path: root.join("runtime/boot_log.jsonl"),
            agent_state_path: root.join("runtime/state/agent_state.json"),
            system_census_path: root.join("runtime/state/system_census.json"),
            browser_engine_state_path: root.join("runtime/state/browser_engine_state.json"),
            runtime_db_path: root.join("runtime/cto_agent.db"),
            attach_socket_path: root.join("runtime/cto-agent.sock"),
            runtime_lock_path: root.join("runtime/cto-agent.lock"),
            pending_hard_reset_report_path: root.join("runtime/recovery/pending-hard-reset-report.json"),
            certs_dir: root.join("runtime/certs"),
            tls_cert_path: root.join("runtime/certs/localhost.crt"),
            tls_key_path: root.join("runtime/certs/localhost.key"),
        };
        let task = sample_task("Keyboard to German", "Use localectl or setxkbmap.");
        let keywords = extract_keywords(&format!("{} {}", task.title, task.detail));
        let mut raw_inclusions = Vec::new();
        append_host_keyboard_inclusions(&paths, &task, &keywords, &mut raw_inclusions);
        assert_eq!(raw_inclusions.len(), 2);
        assert_eq!(raw_inclusions[0].source_kind, "repo_skill");
        assert_eq!(raw_inclusions[1].source_kind, "host_keyboard_contract");
        assert!(raw_inclusions[1].content.contains("keyboard"));
    }
}

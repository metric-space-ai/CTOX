use crate::brain_runtime::grosshirn_runtime_configured;
use crate::brain_runtime::inspect_runtime_disk_headroom;
use crate::brain_runtime::load_kleinhirn_runtime_snapshot;
use crate::brain_runtime::load_runtime_env_map;
use crate::brain_runtime::local_browser_vision_kleinhirn_upgrade_available;
use crate::brain_runtime::local_kleinhirn_upgrade_available;
use crate::command_exec::snapshot_sessions;
use crate::contracts::BootEntry;
use crate::contracts::ContextModePolicy;
use crate::contracts::ContextOptimizationPolicy;
use crate::contracts::ContextQueryToolContract;
use crate::contracts::ExecutionAuthorityPolicy;
use crate::contracts::LoopSafetyPolicy;
use crate::contracts::ModeSystemPolicy;
use crate::contracts::Paths;
use crate::contracts::describe_browser_vision_kleinhirn_selection;
use crate::contracts::describe_kleinhirn_selection;
use crate::contracts::describe_local_kleinhirn_candidates;
use crate::contracts::load_boot_entries;
use crate::contracts::load_census;
use crate::contracts::load_context_governance_policy;
use crate::contracts::load_context_optimization_policy;
use crate::contracts::load_context_policy;
use crate::contracts::load_context_query_tool_contract;
use crate::contracts::load_execution_authority_policy;
use crate::contracts::load_installation_bootstrap_state;
use crate::contracts::load_loop_safety_policy;
use crate::contracts::load_mode_system_policy;
use crate::contracts::load_model_policy;
use crate::contracts::normalize_runtime_model_choice;
use crate::contracts::load_self_preservation_state;
use crate::contracts::now_iso;
use crate::runtime_db::BiosDialogueEntry;
use crate::runtime_db::LearningEntryRecord;
use crate::runtime_db::MemoryItemRecord;
use crate::runtime_db::PersonProfileRecord;
use crate::runtime_db::TaskCheckpointRecord;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::TurnSignalRecord;
use crate::runtime_db::latest_context_package_for_task;
use crate::runtime_db::list_active_learning_entries;
use crate::runtime_db::list_bios_dialogue;
use crate::runtime_db::list_memory_items;
use crate::runtime_db::list_open_tasks;
use crate::runtime_db::list_pending_proactive_contact_candidates;
use crate::runtime_db::list_person_notes_for_person;
use crate::runtime_db::list_person_profiles;
use crate::runtime_db::list_recent_mail_previews_for_person;
use crate::runtime_db::list_recent_task_outcomes;
use crate::runtime_db::list_skills;
use crate::runtime_db::list_task_checkpoints;
use crate::runtime_db::list_turn_signals_for_task;
use crate::runtime_db::load_brain_routing_state;
use crate::runtime_db::load_external_brain_cost_summary;
use crate::runtime_db::load_focus_state;
use crate::runtime_db::load_memory_summary;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::load_task_by_id;
use crate::runtime_db::mark_learning_entries_recalled;
use crate::runtime_db::record_context_package;
use crate::runtime_db::sync_skills;
use reqwest::blocking::Client;
use rusqlite::Connection;
use rusqlite::params;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::env;
use std::fs;
use std::time::Duration;

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
    pub context_optimization: ContextOptimizationSummary,
    pub context_query_contract: ContextQueryContractSummary,
    pub preparation_contract: Option<ContextPreparationContract>,
    pub preparation_review_contract: Option<ContextPreparationReviewContract>,
    pub preparation_state: Option<ContextPreparationState>,
    pub context_query_answers: Vec<ContextQueryAnswer>,
    pub prepared_context_artifact: Option<ContextPreparedArtifact>,
    pub context_distillation: Option<ContextDistilledArtifact>,
    pub compact_controller: Option<ContextCompactControllerEnvelope>,
    pub retrieval_notes: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ContextCompactionTrigger {
    Auto,
    Interrupt,
}

impl ContextCompactionTrigger {
    fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Interrupt => "interrupt",
        }
    }
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

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextOptimizationSummary {
    pub max_questions: usize,
    pub max_matches_per_question: usize,
    pub max_match_chars: usize,
    pub max_total_loops: usize,
    pub required_block_ids: Vec<String>,
    pub block_budgets: Vec<ContextOptimizationBlockBudget>,
    pub surfaces: Vec<crate::contracts::ContextOptimizationSurfacePolicy>,
    pub negative_signals: Vec<crate::contracts::ContextOptimizationSignalPolicy>,
    pub positive_signals: Vec<crate::contracts::ContextOptimizationSignalPolicy>,
    pub assessment_dimensions: Vec<crate::contracts::ContextOptimizationAssessmentDimensionPolicy>,
    pub active_phase: String,
    pub phase_goal: String,
    pub phase_required_outputs: Vec<String>,
    pub phase_allowed_review_decisions: Vec<String>,
    pub go_rule: String,
    pub note_formula: String,
    pub note_bands: Vec<crate::contracts::ContextOptimizationNoteBand>,
    pub note_guardrails: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparationContract {
    pub active_phase: String,
    pub phase_goal: String,
    pub total_max_loops: usize,
    pub phase_max_loops: usize,
    pub required_outputs: Vec<String>,
    pub allowed_review_decisions: Vec<String>,
    pub required_block_ids: Vec<String>,
    pub block_budgets: Vec<ContextOptimizationBlockBudget>,
    pub go_rule: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparationSurfaceCue {
    pub surface_id: String,
    pub title: String,
    pub goal: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparationSignalCue {
    pub signal_id: String,
    pub title: String,
    pub polarity: String,
    pub points: i32,
    pub surface_id: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparationAssessmentCue {
    pub dimension_id: String,
    pub title: String,
    pub weight: usize,
    pub goal: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparationReviewContract {
    pub surfaces: Vec<ContextPreparationSurfaceCue>,
    pub negative_signals: Vec<ContextPreparationSignalCue>,
    pub positive_signals: Vec<ContextPreparationSignalCue>,
    pub assessment_dimensions: Vec<ContextPreparationAssessmentCue>,
    pub note_guardrails: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextQueryContractSummary {
    pub objective: String,
    pub allowed_source_kinds: Vec<String>,
    pub allowed_query_modes: Vec<String>,
    pub default_query_mode: String,
    pub required_question_fields: Vec<String>,
    pub max_questions: usize,
    pub provenance_rule: String,
    pub embedding_search_available: bool,
    pub embedding_search_note: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextOptimizationBlockBudget {
    pub block_id: String,
    pub title: String,
    pub token_budget: usize,
    pub required: bool,
    pub goal: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparationState {
    pub query_rounds_completed: usize,
    pub rewrite_rounds_completed: usize,
    pub review_rounds_completed: usize,
    pub latest_review_decision: String,
    pub latest_review_note: String,
    pub immediate_next_step: String,
    pub missing_evidence: Vec<String>,
    pub weak_blocks: Vec<String>,
    pub repeated_revision_detected: bool,
    pub phase_limit_exceeded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextQueryQuestion {
    #[serde(default)]
    pub question_id: String,
    pub question: String,
    pub why: String,
    #[serde(default)]
    pub query_mode: String,
    #[serde(default)]
    pub source_kinds: Vec<String>,
    #[serde(default)]
    pub max_matches: Option<usize>,
    #[serde(default)]
    pub required_keywords: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextQueryMatch {
    pub source_kind: String,
    pub source_ref: String,
    pub summary: String,
    pub excerpt: String,
    pub score: usize,
    pub query_mode_used: String,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextQueryAnswer {
    pub question: String,
    pub why: String,
    pub matches: Vec<ContextQueryMatch>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparedBlock {
    pub block_id: String,
    pub title: String,
    pub token_budget: usize,
    pub content: String,
    pub why_included: String,
    #[serde(default)]
    pub approx_tokens: Option<usize>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
    #[serde(default)]
    pub omitted_items: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparedFinding {
    pub signal_id: String,
    pub surface_id: String,
    pub polarity: String,
    pub points: i32,
    pub note: String,
    #[serde(default)]
    pub resolution: Option<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparedAssessmentDimension {
    pub dimension_id: String,
    pub note: u8,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparedAssessment {
    pub note: u8,
    pub summary: String,
    #[serde(default)]
    pub strengths: Vec<String>,
    #[serde(default)]
    pub weaknesses: Vec<String>,
    #[serde(default)]
    pub referenced_signal_ids: Vec<String>,
    #[serde(default)]
    pub dimensions: Vec<ContextPreparedAssessmentDimension>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparedReview {
    pub decision: String,
    pub note: String,
    #[serde(default)]
    pub missing_evidence: Vec<String>,
    #[serde(default)]
    pub weak_blocks: Vec<String>,
    #[serde(default)]
    pub budget_violations: Vec<String>,
    #[serde(default)]
    pub repeated_from_prior: bool,
    #[serde(default)]
    pub findings: Vec<ContextPreparedFinding>,
    #[serde(default)]
    pub assessment: Option<ContextPreparedAssessment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPreparedArtifact {
    pub immediate_next_step: String,
    #[serde(default)]
    pub questions: Vec<ContextQueryQuestion>,
    #[serde(default)]
    pub blocks: Vec<ContextPreparedBlock>,
    pub review: ContextPreparedReview,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextDistilledRef {
    pub source_kind: String,
    pub source_ref: String,
    pub label: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextDistilledAnchor {
    pub anchor_id: String,
    pub title: String,
    pub content: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextDistilledFocus {
    pub status: String,
    pub blocker: String,
    pub next_step: String,
    pub done_criteria: String,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextDistilledSnapshot {
    pub summary: String,
    #[serde(default)]
    pub bullet_points: Vec<String>,
    #[serde(default)]
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextContinuityArtifact {
    pub artifact_id: String,
    pub kind: String,
    pub label: String,
    pub summary: String,
    pub source_ref: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextDistilledArtifact {
    pub continuity_narrative: String,
    #[serde(default)]
    pub continuity_artifacts: Vec<ContextContinuityArtifact>,
    #[serde(default)]
    pub continuity_anchors: Vec<ContextDistilledAnchor>,
    #[serde(default)]
    pub system_continuity_anchors: Vec<ContextDistilledAnchor>,
    pub active_focus: ContextDistilledFocus,
    #[serde(default)]
    pub snapshot: Option<ContextDistilledSnapshot>,
    #[serde(default)]
    pub historical_retrieval_refs: Vec<ContextDistilledRef>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCompactProgressReview {
    pub school_grade: u8,
    pub label: String,
    pub summary: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCompactTaskSpawn {
    pub title: String,
    pub detail: String,
    pub priority_bucket: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCompactTaskMutation {
    pub target: String,
    pub action: String,
    pub revised_title: String,
    pub revised_detail: String,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCompactWrapperUpdate {
    pub controller: String,
    pub mode: String,
    pub next_action: String,
    pub active_task: String,
    pub task_packet: Vec<String>,
    pub priority_reason: String,
    pub completed_tasks: Vec<String>,
    pub spawned_tasks: Vec<ContextCompactTaskSpawn>,
    pub mutated_tasks: Vec<ContextCompactTaskMutation>,
    pub priority_order: Vec<String>,
    pub interrupt_triggered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCompactReprioritizationReview {
    pub summary: String,
    pub should_reprioritize: bool,
    pub interrupts_reviewed: bool,
    pub active_task: String,
    pub task_packet: Vec<String>,
    pub priority_reason: String,
    pub next_action: String,
    pub completed_tasks: Vec<String>,
    pub spawned_tasks: Vec<ContextCompactTaskSpawn>,
    pub mutated_tasks: Vec<ContextCompactTaskMutation>,
    pub priority_order: Vec<String>,
    pub wrapper_update: ContextCompactWrapperUpdate,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCompactModelRouting {
    pub tier: String,
    pub current_model: String,
    pub candidate_models: Vec<String>,
    pub requested_model: String,
    pub switch_planned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextCompactControllerEnvelope {
    pub schema_version: String,
    pub controller: String,
    pub trigger: String,
    pub task_hint: String,
    pub progress_review: ContextCompactProgressReview,
    pub continuity_narrative: String,
    pub continuity_anchors: String,
    pub active_focus: String,
    pub reprioritization_review: ContextCompactReprioritizationReview,
    pub model_routing: ContextCompactModelRouting,
}

#[derive(Debug, Clone)]
struct ContextQueryCandidate {
    source_kind: String,
    source_ref: String,
    summary: String,
    text: String,
}

#[derive(Debug, Clone)]
struct RankedQueryHit<'a> {
    candidate: &'a ContextQueryCandidate,
    score: usize,
}

#[derive(Debug, Clone)]
struct ContextEmbeddingTarget {
    base_url: String,
    api_key: String,
    model_id: String,
    chunk_chars: usize,
    chunk_overlap_chars: usize,
    max_batch_size: usize,
    document_instruction: String,
    query_instruction: String,
}

#[derive(Debug, Clone)]
struct CachedEmbeddingChunk {
    source_kind: String,
    source_ref: String,
    chunk_index: usize,
    content_hash: String,
    text_chunk: String,
    embedding: Vec<f32>,
}

fn extract_first_valid_json_value(content: &str) -> Option<serde_json::Value> {
    if let Ok(value) = serde_json::from_str::<serde_json::Value>(content) {
        return Some(value);
    }
    for (start, ch) in content.char_indices() {
        if ch != '{' {
            continue;
        }
        let mut depth = 0_i64;
        let mut in_string = false;
        let mut escaped = false;
        for (offset, current) in content[start..].char_indices() {
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
                            serde_json::from_str::<serde_json::Value>(&content[start..end])
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
}

fn extract_prepared_context_artifact_from_text(text: &str) -> Option<ContextPreparedArtifact> {
    let value = extract_first_valid_json_value(text)?;
    let artifact_value = value.get("preparedContextArtifact").cloned().or_else(|| {
        if value.get("review").is_some() {
            Some(value.clone())
        } else {
            None
        }
    })?;
    serde_json::from_value::<ContextPreparedArtifact>(artifact_value).ok()
}

fn latest_prepared_context_artifact(
    checkpoints: &[TaskCheckpointRecord],
) -> Option<ContextPreparedArtifact> {
    checkpoints
        .iter()
        .find_map(|checkpoint| extract_prepared_context_artifact_from_text(&checkpoint.detail))
}

fn previous_prepared_context_artifact(
    checkpoints: &[TaskCheckpointRecord],
) -> Option<ContextPreparedArtifact> {
    checkpoints
        .iter()
        .find_map(|checkpoint| extract_prepared_context_artifact_from_text(&checkpoint.detail))
}

fn count_checkpoints_with_prefix(checkpoints: &[TaskCheckpointRecord], prefix: &str) -> usize {
    checkpoints
        .iter()
        .filter(|checkpoint| checkpoint.checkpoint_kind.starts_with(prefix))
        .count()
}

fn context_phase_checkpoint_prefix(phase: &str) -> &'static str {
    match phase {
        "query_plan" => "context_query",
        "rewrite" => "context_rewrite",
        "review" => "context_rewrite_review",
        _ => "",
    }
}

fn approx_token_count(text: &str) -> usize {
    let chars = text.chars().count();
    let words = text.split_whitespace().count();
    words.max((chars + 3) / 4)
}

fn normalized_block_text(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_ascii_lowercase()
}

fn checkpoint_contains_machine_evidence(summary: &str, detail: &str) -> bool {
    let summary_lower = summary.to_ascii_lowercase();
    let detail_lower = detail.to_ascii_lowercase();
    if summary_lower.contains("failed")
        || summary_lower.contains("no machine path ran")
        || detail_lower.contains("exec session error:")
        || detail_lower
            .contains("workspace execution contract note: no exec/browser machine path ran")
    {
        return false;
    }
    summary.contains("Bounded command-exec executed:")
        || summary.contains("Started Codex exec session")
        || summary.contains("Wrote to Codex exec session")
        || summary.contains("Read Codex exec session")
        || summary.contains("Bounded command-exec ausgefuehrt:")
        || summary.contains("Exec-Session gestartet")
        || summary.contains("In Codex-Exec-Session geschrieben")
        || summary.contains("Codex-Exec-Session gelesen")
        || detail.contains("Bounded exec result:")
        || detail.contains("Bounded exec session result:")
        || detail.contains("Exec session result:")
}

fn artifacts_repeat_same_blocks(
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
                normalized_block_text(&block.content),
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
                normalized_block_text(&block.content),
            )
        })
        .collect::<Vec<_>>();
    !current_blocks.is_empty() && current_blocks == previous_blocks
}

fn infer_context_optimization_phase(
    task: &TaskRecord,
    artifact: Option<&ContextPreparedArtifact>,
    query_answers: &[ContextQueryAnswer],
) -> String {
    if task.task_kind != "context_preparation" {
        return if artifact.is_some() {
            "prepared_for_execution".to_string()
        } else {
            "execution".to_string()
        };
    }
    let Some(artifact) = artifact else {
        return "query_plan".to_string();
    };
    let decision = artifact.review.decision.trim().to_ascii_lowercase();
    if decision == "go" && !artifact.blocks.is_empty() {
        "ready".to_string()
    } else if artifact.questions.is_empty() {
        "query_plan".to_string()
    } else if query_answers.is_empty() {
        "query_plan".to_string()
    } else if artifact.blocks.is_empty() {
        "rewrite".to_string()
    } else if decision.contains("query") {
        "query_plan".to_string()
    } else {
        "review".to_string()
    }
}

fn prepared_block_by_id<'a>(
    artifact: &'a ContextPreparedArtifact,
    block_id: &str,
) -> Option<&'a ContextPreparedBlock> {
    artifact
        .blocks
        .iter()
        .find(|block| block.block_id == block_id && !block.content.trim().is_empty())
}

fn summarize_raw_inclusion(inclusion: &ContextRawInclusion) -> String {
    let headline = inclusion
        .source_ref
        .rsplit('/')
        .next()
        .unwrap_or(inclusion.source_ref.as_str());
    format!("{} ({headline})", inclusion.source_kind)
}

fn push_unique_line(
    lines: &mut Vec<String>,
    seen: &mut HashSet<String>,
    value: String,
    limit: usize,
) {
    let normalized = normalized_block_text(&value);
    if normalized.is_empty() || !seen.insert(normalized) {
        return;
    }
    if lines.len() < limit {
        lines.push(value);
    }
}

fn infer_source_kind_from_ref(source_ref: &str) -> String {
    let trimmed = source_ref.trim();
    if trimmed.starts_with("task:") {
        if trimmed.matches(':').count() >= 2 {
            "task_checkpoint".to_string()
        } else {
            "task_detail".to_string()
        }
    } else if trimmed.starts_with("learning:") {
        "learning_entry".to_string()
    } else if trimmed.starts_with("memory:") {
        "memory_item".to_string()
    } else if trimmed.starts_with("person:") {
        "person_profile".to_string()
    } else if trimmed.contains(".agents/skills/") || trimmed.ends_with("SKILL.md") {
        "skill".to_string()
    } else if trimmed.starts_with("distillation:focus:") {
        "context_distillation_focus".to_string()
    } else if trimmed.starts_with("distillation:story:") {
        "context_distillation_story".to_string()
    } else if trimmed.starts_with("distillation:snapshot:") {
        "context_distillation_snapshot".to_string()
    } else if trimmed.starts_with("distillation:artifact:") {
        "context_distillation_artifact".to_string()
    } else if trimmed.starts_with("distillation:system:") {
        "context_distillation_system_anchor".to_string()
    } else if trimmed.starts_with("distillation:anchor:") {
        "context_distillation_anchor".to_string()
    } else if trimmed.starts_with("exec-session:") {
        "exec_session".to_string()
    } else {
        "task_detail".to_string()
    }
}

fn infer_ref_label(source_ref: &str) -> String {
    let trimmed = source_ref.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    if let Some(last) = trimmed.rsplit('/').next()
        && last != trimmed
    {
        return last.to_string();
    }
    trimmed.to_string()
}

fn checkpoint_fingerprint(summary: &str, detail: &str) -> String {
    normalized_block_text(&format!("{summary} {detail}"))
}

fn repeated_checkpoint_pair<'a>(
    package: &'a ContextPackage,
) -> Option<(&'a ContextCheckpointEntry, &'a ContextCheckpointEntry)> {
    let latest = package.recent_task_checkpoints.first()?;
    let previous = package.recent_task_checkpoints.get(1)?;
    if checkpoint_fingerprint(&latest.summary, &latest.detail)
        == checkpoint_fingerprint(&previous.summary, &previous.detail)
    {
        Some((latest, previous))
    } else {
        None
    }
}

fn primary_exec_session(package: &ContextPackage) -> Option<&ContextExecSessionEntry> {
    package
        .exec_sessions
        .iter()
        .find(|entry| entry.status == "active")
        .or_else(|| package.exec_sessions.first())
}

fn has_workspace_execution_guidance(package: &ContextPackage) -> bool {
    package.raw_inclusions.iter().any(|item| {
        (item.source_kind == "repo_operation_skill"
            && item.content.contains("Workspace Execution Operations"))
            || (item.source_kind == "system_capability_contract"
                && item
                    .content
                    .to_ascii_lowercase()
                    .contains("repo-local workspace"))
    })
}

fn push_unique_retrieval_ref(
    refs: &mut Vec<ContextDistilledRef>,
    seen: &mut HashSet<(String, String)>,
    source_kind: impl Into<String>,
    source_ref: impl Into<String>,
    label: impl Into<String>,
) {
    let source_kind = source_kind.into();
    let source_ref = source_ref.into();
    if source_kind.trim().is_empty() || source_ref.trim().is_empty() {
        return;
    }
    let key = (source_kind.clone(), source_ref.clone());
    if !seen.insert(key) {
        return;
    }
    refs.push(ContextDistilledRef {
        source_kind,
        source_ref,
        label: trim_chars(&label.into(), 120),
    });
}

fn build_anchor_from_block(block: &ContextPreparedBlock) -> ContextDistilledAnchor {
    ContextDistilledAnchor {
        anchor_id: block.block_id.clone(),
        title: trim_chars(&block.title, 80),
        content: trim_chars(&block.content, 260),
        evidence_refs: block
            .evidence_refs
            .iter()
            .filter(|item| !item.trim().is_empty())
            .take(6)
            .cloned()
            .collect(),
    }
}

fn push_unique_continuity_artifact(
    artifacts: &mut Vec<ContextContinuityArtifact>,
    seen: &mut HashSet<String>,
    kind: impl Into<String>,
    label: impl Into<String>,
    summary: impl Into<String>,
    source_ref: impl Into<String>,
) {
    let kind = kind.into();
    let label = trim_chars(&label.into(), 80);
    let summary = trim_chars(&summary.into(), 220);
    let source_ref = source_ref.into();
    let dedupe_key = format!("{}::{}", kind.trim(), source_ref.trim());
    if kind.trim().is_empty()
        || summary.trim().is_empty()
        || source_ref.trim().is_empty()
        || !seen.insert(dedupe_key)
    {
        return;
    }
    artifacts.push(ContextContinuityArtifact {
        artifact_id: format!("artifact_{}", artifacts.len() + 1),
        kind,
        label,
        summary,
        source_ref,
    });
}

fn build_workstream_continuity_anchors(package: &ContextPackage) -> Vec<ContextDistilledAnchor> {
    if let Some(artifact) = package.prepared_context_artifact.as_ref() {
        let ordered_ids = [
            "goal_and_authority",
            "definition_of_done",
            "relevant_artifacts",
            "constraints_and_policies",
            "risks_and_failure_modes",
            "open_questions",
        ];
        let mut anchors = ordered_ids
            .iter()
            .filter_map(|block_id| prepared_block_by_id(artifact, block_id))
            .map(build_anchor_from_block)
            .collect::<Vec<_>>();
        if anchors.is_empty() {
            anchors = artifact
                .blocks
                .iter()
                .filter(|block| !block.content.trim().is_empty())
                .take(6)
                .map(build_anchor_from_block)
                .collect();
        }
        return anchors;
    }

    let mut anchors = Vec::new();
    anchors.push(ContextDistilledAnchor {
        anchor_id: "task_objective".to_string(),
        title: "Task Objective".to_string(),
        content: trim_chars(
            &format!("{} {}", package.task_brief.title, package.task_brief.detail),
            260,
        ),
        evidence_refs: vec![format!("task:{}", package.task_id)],
    });
    if let Some(checkpoint) = package.recent_task_checkpoints.first() {
        anchors.push(ContextDistilledAnchor {
            anchor_id: "latest_checkpoint".to_string(),
            title: "Latest Checkpoint".to_string(),
            content: trim_chars(
                &format!("{} {}", checkpoint.summary, checkpoint.detail),
                260,
            ),
            evidence_refs: vec![format!(
                "task:{}:{}",
                package.task_id, checkpoint.checkpoint_kind
            )],
        });
    }
    if !package.raw_inclusions.is_empty() {
        let items = package
            .raw_inclusions
            .iter()
            .take(4)
            .map(summarize_raw_inclusion)
            .collect::<Vec<_>>()
            .join(", ");
        anchors.push(ContextDistilledAnchor {
            anchor_id: "relevant_artifacts".to_string(),
            title: "Relevant Artifacts".to_string(),
            content: trim_chars(&items, 240),
            evidence_refs: package
                .raw_inclusions
                .iter()
                .take(4)
                .map(|item| item.source_ref.clone())
                .collect(),
        });
    }
    if let Some(session) = primary_exec_session(package) {
        anchors.push(ContextDistilledAnchor {
            anchor_id: "workspace_continuity".to_string(),
            title: "Workspace Continuity".to_string(),
            content: trim_chars(
                &format!(
                    "Exec session {} ({}) is available for this task and should be reused before another one-shot replay.",
                    session.session_id, session.status
                ),
                240,
            ),
            evidence_refs: vec![format!("task:{}", package.task_id)],
        });
    }
    if !package.relevant_learning_entries.is_empty() {
        let learning = package
            .relevant_learning_entries
            .iter()
            .take(2)
            .map(|item| item.summary.clone())
            .collect::<Vec<_>>()
            .join(" ");
        anchors.push(ContextDistilledAnchor {
            anchor_id: "relevant_learning".to_string(),
            title: "Relevant Learning".to_string(),
            content: trim_chars(&learning, 220),
            evidence_refs: package
                .relevant_learning_entries
                .iter()
                .take(2)
                .map(|item| format!("learning:{}", item.id))
                .collect(),
        });
    }
    anchors
}

fn build_continuity_artifacts(package: &ContextPackage) -> Vec<ContextContinuityArtifact> {
    let mut artifacts = Vec::new();
    let mut seen = HashSet::new();

    if let Some(artifact) = package.prepared_context_artifact.as_ref()
        && let Some(block) = prepared_block_by_id(artifact, "relevant_artifacts")
    {
        let source_ref = block
            .evidence_refs
            .first()
            .cloned()
            .unwrap_or_else(|| format!("prepared:{}:relevant_artifacts", package.task_id));
        push_unique_continuity_artifact(
            &mut artifacts,
            &mut seen,
            "prepared_block",
            "Prepared Relevant Artifacts",
            &block.content,
            source_ref,
        );
    }

    if let Some(checkpoint) = package.recent_task_checkpoints.first() {
        push_unique_continuity_artifact(
            &mut artifacts,
            &mut seen,
            "checkpoint",
            format!("Latest Checkpoint ({})", checkpoint.checkpoint_kind),
            format!("{} {}", checkpoint.summary, checkpoint.detail),
            format!("task:{}:{}", package.task_id, checkpoint.checkpoint_kind),
        );
    }

    if let Some(session) = primary_exec_session(package) {
        push_unique_continuity_artifact(
            &mut artifacts,
            &mut seen,
            "exec_session",
            format!("Exec Session {}", trim_chars(&session.session_id, 48)),
            format!(
                "Reuse session {} in {} status from {}.",
                trim_chars(&session.session_id, 48),
                session.status,
                trim_chars(&session.cwd, 120)
            ),
            format!("exec-session:{}", session.session_id),
        );
    }

    for inclusion in package
        .raw_inclusions
        .iter()
        .filter(|item| {
            matches!(
                item.source_kind.as_str(),
                "current_task_machine_evidence"
                    | "system_capability_contract"
                    | "repo_operation_skill"
                    | "definition_of_done_policy"
                    | "installation_tool_smoke_resource"
                    | "task_detail"
                    | "parent_task"
            )
        })
        .take(5)
    {
        let label = match inclusion.source_kind.as_str() {
            "current_task_machine_evidence" => "Machine Evidence",
            "system_capability_contract" => "Capability Contract",
            "repo_operation_skill" => "Repo Skill",
            "definition_of_done_policy" => "Definition Of Done",
            "installation_tool_smoke_resource" => "Smoke Resource",
            "parent_task" => "Parent Task",
            _ => "Task Artifact",
        };
        push_unique_continuity_artifact(
            &mut artifacts,
            &mut seen,
            inclusion.source_kind.clone(),
            label,
            summarize_raw_inclusion(inclusion),
            inclusion.source_ref.clone(),
        );
    }

    artifacts.truncate(6);
    artifacts
}

fn build_system_continuity_anchors(package: &ContextPackage) -> Vec<ContextDistilledAnchor> {
    vec![
        ContextDistilledAnchor {
            anchor_id: "priority_and_authority".to_string(),
            title: "Priority And Authority".to_string(),
            content: trim_chars(
                &format!(
                    "{} {}",
                    package.loop_safety.priority_law, package.execution_authority.escalation_rule
                ),
                260,
            ),
            evidence_refs: vec![
                format!(
                    "distillation:system:{}:priority_and_authority",
                    package.task_id
                ),
                format!("task:{}", package.task_id),
            ],
        },
        ContextDistilledAnchor {
            anchor_id: "context_governance".to_string(),
            title: "Context Governance".to_string(),
            content: trim_chars(
                &format!(
                    "{} {}",
                    package.context_governance.normal_context_control,
                    package.context_governance.emergency_boundary
                ),
                260,
            ),
            evidence_refs: vec![
                format!("distillation:system:{}:context_governance", package.task_id),
                format!("task:{}", package.task_id),
            ],
        },
        ContextDistilledAnchor {
            anchor_id: "skill_surface".to_string(),
            title: "Skill Surface".to_string(),
            content: trim_chars(
                &format!(
                    "{} {}",
                    package.skill_system.sync_rule, package.skill_system.operating_rule
                ),
                260,
            ),
            evidence_refs: vec![
                format!("distillation:system:{}:skill_surface", package.task_id),
                format!("task:{}", package.task_id),
            ],
        },
        ContextDistilledAnchor {
            anchor_id: "host_and_brain_guard".to_string(),
            title: "Host And Brain Guard".to_string(),
            content: trim_chars(
                &format!(
                    "{} {}",
                    package.host_survival.operating_rule, package.brain_access.operating_rule
                ),
                260,
            ),
            evidence_refs: vec![
                format!(
                    "distillation:system:{}:host_and_brain_guard",
                    package.task_id
                ),
                format!("task:{}", package.task_id),
            ],
        },
    ]
    .into_iter()
    .filter(|anchor| !anchor.content.trim().is_empty())
    .collect()
}

fn build_continuity_narrative(package: &ContextPackage) -> String {
    if let Some(artifact) = package.prepared_context_artifact.as_ref() {
        let mut parts = Vec::new();
        if let Some(goal) = prepared_block_by_id(artifact, "goal_and_authority") {
            parts.push(trim_chars(&goal.content, 220));
        }
        if let Some(world) = prepared_block_by_id(artifact, "verified_world_state") {
            parts.push(trim_chars(&world.content, 420));
        }
        if !artifact.review.note.trim().is_empty() {
            parts.push(format!(
                "Preparation review: {}",
                trim_chars(&artifact.review.note, 180)
            ));
        }
        if let Some(constraints) = prepared_block_by_id(artifact, "constraints_and_policies") {
            parts.push(format!(
                "Persistent constraints: {}",
                trim_chars(&constraints.content, 180)
            ));
        }
        return trim_chars(&parts.join(" "), 900);
    }

    let mut parts = vec![trim_chars(
        &format!(
            "Current task: {} {}",
            package.task_brief.title, package.task_brief.detail
        ),
        320,
    )];
    if let Some(checkpoint) = package.recent_task_checkpoints.first() {
        parts.push(format!(
            "Latest verified checkpoint: {} {}",
            trim_chars(&checkpoint.summary, 180),
            trim_chars(&checkpoint.detail, 200)
        ));
    }
    if let Some(anchor) = package
        .raw_inclusions
        .iter()
        .find(|item| item.source_kind == "current_task_machine_evidence")
    {
        parts.push(format!(
            "Verified workspace anchor: {}",
            trim_chars(&anchor.content, 220)
        ));
    }
    if repeated_checkpoint_pair(package).is_some() {
        parts.push(
            "The latest two checkpoints describe the same bounded step, so the next turn must continue from current workspace state instead of replaying the same command."
                .to_string(),
        );
    }
    if let Some(session) = primary_exec_session(package) {
        parts.push(format!(
            "Exec session continuity is available through {} ({}).",
            trim_chars(&session.session_id, 80),
            session.status
        ));
    }
    if let Some(learning) = package.relevant_learning_entries.first() {
        parts.push(format!(
            "Relevant prior learning: {}",
            trim_chars(&learning.summary, 180)
        ));
    }
    trim_chars(&parts.join(" "), 900)
}

fn build_active_focus(package: &ContextPackage) -> ContextDistilledFocus {
    if let Some(artifact) = package.prepared_context_artifact.as_ref() {
        let status = prepared_block_by_id(artifact, "verified_world_state")
            .map(|block| trim_chars(&block.content, 220))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| trim_chars(&artifact.review.note, 220));
        let blocker = if !artifact.review.missing_evidence.is_empty() {
            trim_chars(&artifact.review.missing_evidence.join("; "), 220)
        } else if let Some(block) = prepared_block_by_id(artifact, "open_questions") {
            trim_chars(&block.content, 220)
        } else if let Some(block) = prepared_block_by_id(artifact, "risks_and_failure_modes") {
            trim_chars(&block.content, 220)
        } else {
            String::new()
        };
        let next_step = if !artifact.immediate_next_step.trim().is_empty() {
            trim_chars(&artifact.immediate_next_step, 220)
        } else {
            prepared_block_by_id(artifact, "next_action_only")
                .map(|block| trim_chars(&block.content, 220))
                .unwrap_or_default()
        };
        let done_criteria = prepared_block_by_id(artifact, "definition_of_done")
            .map(|block| trim_chars(&block.content, 220))
            .unwrap_or_default();
        let mut evidence_refs = Vec::new();
        if let Some(block) = prepared_block_by_id(artifact, "verified_world_state") {
            evidence_refs.extend(block.evidence_refs.iter().take(4).cloned());
        }
        if let Some(block) = prepared_block_by_id(artifact, "next_action_only") {
            evidence_refs.extend(block.evidence_refs.iter().take(4).cloned());
        }
        if let Some(block) = prepared_block_by_id(artifact, "definition_of_done") {
            evidence_refs.extend(block.evidence_refs.iter().take(4).cloned());
        }
        evidence_refs.retain(|item| !item.trim().is_empty());
        evidence_refs.truncate(8);
        return ContextDistilledFocus {
            status,
            blocker,
            next_step,
            done_criteria,
            evidence_refs,
        };
    }

    let repeated_pair = repeated_checkpoint_pair(package);
    let primary_session = primary_exec_session(package);
    let workspace_guidance = has_workspace_execution_guidance(package);
    let workspace_anchor = package
        .raw_inclusions
        .iter()
        .find(|item| item.source_kind == "current_task_machine_evidence");
    let needs_session_or_targeted_step =
        workspace_guidance && workspace_anchor.is_some() && primary_session.is_none();
    let status = if let Some((latest, _)) = repeated_pair {
        trim_chars(
            &format!("Recent bounded step is repeating: {}", latest.summary),
            220,
        )
    } else {
        package
            .recent_task_checkpoints
            .first()
            .map(|item| trim_chars(&item.summary, 220))
            .filter(|value| !value.is_empty())
            .unwrap_or_else(|| trim_chars(&package.task_brief.detail, 220))
    };
    let blocker = if let Some((latest, _)) = repeated_pair {
        let mut message = format!(
            "The latest two checkpoints collapse to the same bounded step. Latest: {}",
            trim_chars(&latest.summary, 140)
        );
        if workspace_anchor.is_some() {
            message.push_str(
                " The prior turn already captured a verified workspace anchor, so the next step must continue from that machine evidence instead of claiming the context is missing again.",
            );
        }
        if let Some(session) = primary_session {
            message.push_str(&format!(
                " Reuse exec session {} ({}) instead of another one-shot replay.",
                trim_chars(&session.session_id, 48),
                session.status
            ));
        } else if latest.detail.contains("Bounded command-exec executed:") {
            message.push_str(" Another one-shot command replay is unlikely to add new evidence.");
        }
        trim_chars(&message, 220)
    } else if needs_session_or_targeted_step {
        "A verified workspace anchor already exists for this repo task, but no exec session continuity is visible yet. Continue from that anchor with either a task-bound exec session or one exact anchored machine step; another broad repo scan or history reload is not a valid continuation unless the missing fact is named explicitly.".to_string()
    } else {
        package
            .recent_turn_signals
            .first()
            .map(|item| trim_chars(&item.message, 200))
            .unwrap_or_default()
    };
    let next_step = if let Some(session) = primary_session {
        if repeated_pair.is_some() {
            trim_chars(
                &format!(
                    "Reuse exec session {} and inspect current workspace state/output before issuing another write, edit, or build command.",
                    session.session_id
                ),
                220,
            )
        } else {
            trim_chars(
                &format!(
                    "Continue task #{}, reusing exec session {} if the work stays multi-step.",
                    package.task_id, session.session_id
                ),
                220,
            )
        }
    } else if needs_session_or_targeted_step {
        "Start a task-bound exec session now, or run one exact anchored machine step against the verified workspace path/file/build target. Do not spend this turn on another broad repo scan or broad history reload unless you name the exact missing fact that the anchor still does not provide.".to_string()
    } else if workspace_anchor.is_some() {
        "Use the verified workspace anchor from the last machine step to take one concrete machine action now: targeted file inspection, edit, build, or test. Do not reopen broad history reload or claim completion without a machine path.".to_string()
    } else if repeated_pair.is_some() {
        "Inspect the current workspace state and diff from the repeated bounded step before issuing another command.".to_string()
    } else {
        trim_chars(
            &format!(
                "Advance task #{}: {}",
                package.task_id, package.task_brief.title
            ),
            200,
        )
    };
    let mut evidence_refs = vec![format!("task:{}", package.task_id)];
    if let Some((latest, previous)) = repeated_pair {
        evidence_refs.push(format!(
            "task:{}:{}",
            package.task_id, latest.checkpoint_kind
        ));
        evidence_refs.push(format!(
            "task:{}:{}",
            package.task_id, previous.checkpoint_kind
        ));
    } else if let Some(item) = package.recent_task_checkpoints.first() {
        evidence_refs.push(format!("task:{}:{}", package.task_id, item.checkpoint_kind));
    }
    if let Some(anchor) = workspace_anchor {
        evidence_refs.push(anchor.source_ref.clone());
    }
    evidence_refs.sort();
    evidence_refs.dedup();
    ContextDistilledFocus {
        status,
        blocker,
        next_step,
        done_criteria: package
            .raw_inclusions
            .iter()
            .find(|item| item.source_kind == "definition_of_done_policy")
            .map(|item| trim_chars(&item.content, 220))
            .unwrap_or_else(|| {
                "Produce fresh verified progress for the active task and keep the current constraints intact."
                    .to_string()
            }),
        evidence_refs,
    }
}

fn build_context_snapshot(package: &ContextPackage) -> Option<ContextDistilledSnapshot> {
    let summary = package
        .recent_task_checkpoints
        .first()
        .map(|item| trim_chars(&item.summary, 220))
        .filter(|value| !value.is_empty())
        .or_else(|| {
            if package.focus_state.note.trim().is_empty() {
                None
            } else {
                Some(trim_chars(&package.focus_state.note, 220))
            }
        })?;
    let mut bullets = Vec::new();
    let mut seen = HashSet::new();
    push_unique_line(
        &mut bullets,
        &mut seen,
        format!("Context mode: {}", package.context_mode),
        6,
    );
    push_unique_line(
        &mut bullets,
        &mut seen,
        format!("Agent mode: {}", package.current_agent_mode),
        6,
    );
    push_unique_line(
        &mut bullets,
        &mut seen,
        format!("Queue depth: {}", package.focus_state.queue_depth),
        6,
    );
    if let Some(checkpoint) = package.recent_task_checkpoints.first() {
        push_unique_line(
            &mut bullets,
            &mut seen,
            format!("Latest checkpoint kind: {}", checkpoint.checkpoint_kind),
            6,
        );
    }
    if repeated_checkpoint_pair(package).is_some() {
        push_unique_line(
            &mut bullets,
            &mut seen,
            "Repeated checkpoint detected: the latest two bounded steps match.".to_string(),
            6,
        );
    }
    if let Some(anchor) = package
        .raw_inclusions
        .iter()
        .find(|item| item.source_kind == "current_task_machine_evidence")
    {
        push_unique_line(
            &mut bullets,
            &mut seen,
            format!(
                "Verified workspace anchor: {}",
                trim_chars(&anchor.content, 180)
            ),
            6,
        );
    }
    if let Some(session) = primary_exec_session(package) {
        push_unique_line(
            &mut bullets,
            &mut seen,
            format!(
                "Exec session: {} ({})",
                trim_chars(&session.session_id, 48),
                session.status
            ),
            6,
        );
    }
    if let Some(state) = package.preparation_state.as_ref()
        && !state.latest_review_decision.trim().is_empty()
    {
        push_unique_line(
            &mut bullets,
            &mut seen,
            format!("Preparation decision: {}", state.latest_review_decision),
            6,
        );
    }
    Some(ContextDistilledSnapshot {
        summary,
        bullet_points: bullets,
        evidence_refs: package
            .recent_task_checkpoints
            .iter()
            .take(2)
            .map(|item| format!("task:{}:{}", package.task_id, item.checkpoint_kind))
            .collect(),
    })
}

fn build_historical_retrieval_refs(
    package: &ContextPackage,
    continuity_artifacts: &[ContextContinuityArtifact],
    anchors: &[ContextDistilledAnchor],
    system_anchors: &[ContextDistilledAnchor],
    focus: &ContextDistilledFocus,
    snapshot: Option<&ContextDistilledSnapshot>,
) -> Vec<ContextDistilledRef> {
    let mut refs = Vec::new();
    let mut seen = HashSet::new();
    push_unique_retrieval_ref(
        &mut refs,
        &mut seen,
        "task_detail",
        format!("task:{}", package.task_id),
        format!("Task #{} {}", package.task_id, package.task_title),
    );
    for checkpoint in package.recent_task_checkpoints.iter().take(3) {
        push_unique_retrieval_ref(
            &mut refs,
            &mut seen,
            "task_checkpoint",
            format!("task:{}:{}", package.task_id, checkpoint.checkpoint_kind),
            checkpoint.summary.clone(),
        );
    }
    push_unique_retrieval_ref(
        &mut refs,
        &mut seen,
        "context_distillation_story",
        format!("distillation:story:{}", package.task_id),
        "Continuity narrative".to_string(),
    );
    push_unique_retrieval_ref(
        &mut refs,
        &mut seen,
        "context_distillation_focus",
        format!("distillation:focus:{}", package.task_id),
        "Active focus".to_string(),
    );
    if snapshot.is_some() {
        push_unique_retrieval_ref(
            &mut refs,
            &mut seen,
            "context_distillation_snapshot",
            format!("distillation:snapshot:{}", package.task_id),
            "Context snapshot".to_string(),
        );
    }
    for artifact in continuity_artifacts {
        push_unique_retrieval_ref(
            &mut refs,
            &mut seen,
            "context_distillation_artifact",
            format!(
                "distillation:artifact:{}:{}",
                package.task_id, artifact.artifact_id
            ),
            artifact.label.clone(),
        );
        push_unique_retrieval_ref(
            &mut refs,
            &mut seen,
            infer_source_kind_from_ref(&artifact.source_ref),
            artifact.source_ref.clone(),
            infer_ref_label(&artifact.source_ref),
        );
    }
    for anchor in anchors {
        push_unique_retrieval_ref(
            &mut refs,
            &mut seen,
            "context_distillation_anchor",
            format!(
                "distillation:anchor:{}:{}",
                package.task_id, anchor.anchor_id
            ),
            anchor.title.clone(),
        );
        for evidence_ref in &anchor.evidence_refs {
            push_unique_retrieval_ref(
                &mut refs,
                &mut seen,
                infer_source_kind_from_ref(evidence_ref),
                evidence_ref.clone(),
                infer_ref_label(evidence_ref),
            );
        }
    }
    for anchor in system_anchors {
        push_unique_retrieval_ref(
            &mut refs,
            &mut seen,
            "context_distillation_system_anchor",
            format!(
                "distillation:system:{}:{}",
                package.task_id, anchor.anchor_id
            ),
            anchor.title.clone(),
        );
    }
    for evidence_ref in &focus.evidence_refs {
        push_unique_retrieval_ref(
            &mut refs,
            &mut seen,
            infer_source_kind_from_ref(evidence_ref),
            evidence_ref.clone(),
            infer_ref_label(evidence_ref),
        );
    }
    if let Some(snapshot) = snapshot {
        for evidence_ref in &snapshot.evidence_refs {
            push_unique_retrieval_ref(
                &mut refs,
                &mut seen,
                infer_source_kind_from_ref(evidence_ref),
                evidence_ref.clone(),
                infer_ref_label(evidence_ref),
            );
        }
    }
    refs.truncate(18);
    refs
}

fn build_context_distillation(package: &ContextPackage) -> Option<ContextDistilledArtifact> {
    let continuity_narrative = build_continuity_narrative(package);
    let continuity_artifacts = build_continuity_artifacts(package);
    let continuity_anchors = build_workstream_continuity_anchors(package);
    let system_continuity_anchors = build_system_continuity_anchors(package);
    let active_focus = build_active_focus(package);
    let snapshot = build_context_snapshot(package);
    let historical_retrieval_refs = build_historical_retrieval_refs(
        package,
        &continuity_artifacts,
        &continuity_anchors,
        &system_continuity_anchors,
        &active_focus,
        snapshot.as_ref(),
    );

    if continuity_narrative.trim().is_empty()
        && continuity_artifacts.is_empty()
        && continuity_anchors.is_empty()
        && system_continuity_anchors.is_empty()
        && active_focus.status.trim().is_empty()
        && active_focus.next_step.trim().is_empty()
    {
        return None;
    }

    Some(ContextDistilledArtifact {
        continuity_narrative,
        continuity_artifacts,
        continuity_anchors,
        system_continuity_anchors,
        active_focus,
        snapshot,
        historical_retrieval_refs,
    })
}

fn build_compact_controller(
    paths: &Paths,
    task: &TaskRecord,
    package: &ContextPackage,
    trigger: ContextCompactionTrigger,
) -> Option<ContextCompactControllerEnvelope> {
    let distillation = package.context_distillation.as_ref()?;
    let open_tasks = list_open_tasks(paths, 8).unwrap_or_default();
    let (school_grade, progress_summary, progress_rationale) =
        evaluate_progress_school_grade(task, package, trigger, &open_tasks);
    let model_routing = select_compact_model_routing(paths, package, school_grade);
    let reprioritization_review =
        build_reprioritization_review(task, package, trigger, &open_tasks);
    Some(ContextCompactControllerEnvelope {
        schema_version: "compact_controller_v1".to_string(),
        controller: "context_optimizer_simple_html_root_v1".to_string(),
        trigger: trigger.as_str().to_string(),
        task_hint: build_compact_task_hint(task),
        progress_review: ContextCompactProgressReview {
            school_grade,
            label: compact_grade_label(school_grade).to_string(),
            summary: progress_summary,
            rationale: progress_rationale,
        },
        continuity_narrative: trim_chars(&distillation.continuity_narrative, 900),
        continuity_anchors: render_compact_continuity_anchors(distillation),
        active_focus: render_compact_active_focus(distillation),
        reprioritization_review,
        model_routing,
    })
}

fn build_compact_task_hint(task: &TaskRecord) -> String {
    let detail = trim_chars(&task.detail, 280);
    if detail.is_empty() {
        task.title.trim().to_string()
    } else {
        format!("{}\n\n{}", task.title.trim(), detail)
    }
}

fn compact_grade_label(grade: u8) -> &'static str {
    match grade {
        1 => "sehr_gut",
        2 => "gut",
        3 => "befriedigend",
        4 => "ausreichend",
        5 => "mangelhaft",
        _ => "ungenuegend",
    }
}

fn evaluate_progress_school_grade(
    task: &TaskRecord,
    package: &ContextPackage,
    trigger: ContextCompactionTrigger,
    open_tasks: &[TaskRecord],
) -> (u8, String, String) {
    let mut grade = 2_u8;
    let mut reasons = Vec::new();

    if trigger == ContextCompactionTrigger::Interrupt {
        grade = grade.max(3);
        reasons.push(
            "An interrupt arrived, so continuity and reprioritization both matter now."
                .to_string(),
        );
    }

    if !package.recent_turn_signals.is_empty() {
        grade = grade.max(3);
        reasons.push("Recent turn signals show live interruption pressure.".to_string());
    }

    if task.run_count >= 5 {
        grade = grade.max(5);
        reasons.push("The task already consumed many bounded runs without final closure.".to_string());
    } else if task.run_count >= 3 {
        grade = grade.max(4);
        reasons.push("The task already needed several bounded runs.".to_string());
    }

    if let Some(preparation_state) = package.preparation_state.as_ref() {
        if !preparation_state.missing_evidence.is_empty() {
            grade = grade.max(4);
            reasons.push("The current context still reports missing evidence.".to_string());
        }
        if !preparation_state.weak_blocks.is_empty() || preparation_state.repeated_revision_detected {
            grade = grade.max(4);
            reasons.push("Weak or repeated context blocks indicate unstable progress.".to_string());
        }
        if preparation_state.phase_limit_exceeded {
            grade = grade.max(5);
            reasons.push("The preparation phase exceeded its safe iteration budget.".to_string());
        }
        if preparation_state.latest_review_decision.eq_ignore_ascii_case("blocked") {
            grade = grade.max(5);
            reasons.push("The latest preparation review is blocked.".to_string());
        } else if preparation_state.latest_review_decision.eq_ignore_ascii_case("go") {
            grade = grade.min(2);
        }
    }

    if let Some(artifact) = package.prepared_context_artifact.as_ref() {
        if artifact.review.decision.eq_ignore_ascii_case("blocked") {
            grade = grade.max(5);
            reasons.push("The prepared context artifact is explicitly blocked.".to_string());
        } else if artifact.review.decision.eq_ignore_ascii_case("go") && task.run_count <= 1 {
            grade = grade.min(2);
        }
        if !artifact.review.missing_evidence.is_empty() || !artifact.review.weak_blocks.is_empty() {
            grade = grade.max(4);
        }
    }

    if let Some(distillation) = package.context_distillation.as_ref() {
        if !distillation.active_focus.blocker.trim().is_empty() {
            grade = grade.max(4);
            reasons.push("The active focus still names a real blocker.".to_string());
        }
    }

    if open_tasks
        .iter()
        .any(|queued| queued.id != task.id && queued.task_kind == "owner_interrupt")
        && task.task_kind != "owner_interrupt"
    {
        grade = grade.max(4);
        reasons.push("A queued owner interrupt competes with the current task.".to_string());
    }

    let summary = match grade {
        1 => "The agent is moving with very strong continuity and little visible friction.",
        2 => "The agent is progressing well and only needs light compaction for continuity.",
        3 => "The agent is progressing, but the workstream is no longer trivial and needs explicit steering.",
        4 => "Progress is only adequate; compaction should actively stabilize focus and task order.",
        5 => "Progress is weak; the loop should assume reprioritization and stronger model help may be necessary.",
        _ => "Progress is critical; the current workstream needs aggressive stabilization and likely escalation.",
    }
    .to_string();

    if reasons.is_empty() {
        reasons.push("No acute continuity risk beyond normal bounded work was detected.".to_string());
    }

    (grade.clamp(1, 6), summary, reasons.join(" "))
}

fn select_compact_model_routing(
    paths: &Paths,
    package: &ContextPackage,
    school_grade: u8,
) -> ContextCompactModelRouting {
    let env_map = load_runtime_env_map(paths).unwrap_or_default();
    let simple_candidates = compact_candidate_models(
        env_map.get("CTO_AGENT_COMPACT_SIMPLE_MODEL"),
        &["openai/gpt-oss-20b", "openai/gpt-5.4-nano"],
    );
    let medium_candidates = compact_candidate_models(
        env_map.get("CTO_AGENT_COMPACT_MEDIUM_MODEL"),
        &["openai/gpt-5.4-nano", "openai/gpt-5.4-mini"],
    );
    let red_candidates = compact_candidate_models(
        env_map.get("CTO_AGENT_COMPACT_RED_MODEL"),
        &["openai/gpt-5.4-mini", "openai/gpt-5.4"],
    );
    let (tier, candidate_models) = match school_grade {
        1 | 2 => ("simple", simple_candidates),
        3 | 4 => ("medium", medium_candidates),
        _ => ("red", red_candidates),
    };
    let current_model = if package.brain_access.active_brain_route == "grosshirn" {
        first_non_empty_owned(&[
            env_map.get("CTO_AGENT_GROSSHIRN_MODEL").cloned(),
            package.brain_access.grosshirn_candidates.first().cloned(),
        ])
        .unwrap_or_else(|| "openai/gpt-5.4".to_string())
    } else {
        first_non_empty_owned(&[
            env_map.get("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL").cloned(),
            env_map.get("CTO_AGENT_KLEINHIRN_MODEL").cloned(),
            Some(package.brain_access.local_selected_kleinhirn.clone()),
            Some(package.brain_access.current_runtime_kleinhirn.clone()),
        ])
        .unwrap_or_else(|| "openai/gpt-oss-20b".to_string())
    };
    let requested_model = candidate_models
        .first()
        .cloned()
        .unwrap_or_else(|| current_model.clone());
    ContextCompactModelRouting {
        tier: tier.to_string(),
        current_model: current_model.clone(),
        candidate_models: candidate_models.clone(),
        requested_model: requested_model.clone(),
        switch_planned: normalize_for_compare(&current_model)
            != normalize_for_compare(&requested_model),
    }
}

fn compact_candidate_models(value: Option<&String>, defaults: &[&str]) -> Vec<String> {
    if let Some(raw) = value {
        let parsed = raw
            .split(',')
            .map(|item| item.trim())
            .filter(|item| !item.is_empty())
            .map(normalize_runtime_model_choice)
            .collect::<Vec<_>>();
        if !parsed.is_empty() {
            return parsed;
        }
    }
    defaults
        .iter()
        .map(|item| normalize_runtime_model_choice(item))
        .collect()
}

fn first_non_empty_owned(values: &[Option<String>]) -> Option<String> {
    values.iter().find_map(|value| {
        value
            .as_ref()
            .map(|item| item.trim())
            .filter(|item| !item.is_empty())
            .map(ToString::to_string)
    })
}

fn normalize_for_compare(value: &str) -> String {
    value.trim().to_ascii_lowercase()
}

fn build_reprioritization_review(
    task: &TaskRecord,
    package: &ContextPackage,
    trigger: ContextCompactionTrigger,
    open_tasks: &[TaskRecord],
) -> ContextCompactReprioritizationReview {
    let active_task = format!("#{} {}", task.id, trim_chars(&task.title, 120));
    let mut task_packet = vec![active_task.clone()];
    let mut priority_order = vec![format!("#{}", task.id)];
    for queued in open_tasks.iter().filter(|queued| queued.id != task.id).take(4) {
        task_packet.push(format!("#{} {}", queued.id, trim_chars(&queued.title, 120)));
        priority_order.push(format!("#{}", queued.id));
    }

    let owner_interrupt_waiting = open_tasks
        .iter()
        .any(|queued| queued.id != task.id && queued.task_kind == "owner_interrupt");
    let interrupts_reviewed = trigger == ContextCompactionTrigger::Interrupt
        || !package.recent_turn_signals.is_empty();
    let should_reprioritize = interrupts_reviewed || owner_interrupt_waiting;
    let next_action = if should_reprioritize {
        "reprioritize"
    } else {
        "continue_current_task"
    };
    let priority_reason = if owner_interrupt_waiting {
        "A queued owner interrupt is waiting and should be reconsidered against the current task packet."
            .to_string()
    } else if interrupts_reviewed {
        "The compaction was triggered by or near an interrupt, so the wrapper should review task order now."
            .to_string()
    } else {
        "No stronger competing task packet is visible right now.".to_string()
    };
    let summary = if should_reprioritize {
        "Compaction recommends a task-order review before the next bounded step.".to_string()
    } else {
        "Compaction keeps the current workstream in place.".to_string()
    };

    ContextCompactReprioritizationReview {
        summary,
        should_reprioritize,
        interrupts_reviewed,
        active_task: active_task.clone(),
        task_packet: task_packet.clone(),
        priority_reason: priority_reason.clone(),
        next_action: next_action.to_string(),
        completed_tasks: Vec::new(),
        spawned_tasks: Vec::new(),
        mutated_tasks: Vec::new(),
        priority_order: priority_order.clone(),
        wrapper_update: ContextCompactWrapperUpdate {
            controller: "compact_controller".to_string(),
            mode: if should_reprioritize {
                "reprioritize".to_string()
            } else {
                "execute_task".to_string()
            },
            next_action: next_action.to_string(),
            active_task,
            task_packet,
            priority_reason,
            completed_tasks: Vec::new(),
            spawned_tasks: Vec::new(),
            mutated_tasks: Vec::new(),
            priority_order,
            interrupt_triggered: trigger == ContextCompactionTrigger::Interrupt,
        },
    }
}

fn render_compact_continuity_anchors(distillation: &ContextDistilledArtifact) -> String {
    let mut fragments = Vec::new();
    for artifact in distillation.continuity_artifacts.iter().take(6) {
        fragments.push(format!(
            "{} [{}]: {}",
            artifact.label,
            artifact.kind,
            trim_chars(&artifact.summary, 160)
        ));
    }
    for anchor in distillation.continuity_anchors.iter().take(6) {
        fragments.push(format!(
            "{}: {}",
            anchor.title,
            trim_chars(&anchor.content, 180)
        ));
    }
    for anchor in distillation.system_continuity_anchors.iter().take(4) {
        fragments.push(format!(
            "system {}: {}",
            anchor.title,
            trim_chars(&anchor.content, 160)
        ));
    }
    trim_chars(&fragments.join("\n"), 1600)
}

fn render_compact_active_focus(distillation: &ContextDistilledArtifact) -> String {
    let focus = &distillation.active_focus;
    trim_chars(
        &format!(
            "Status: {}\nBlocker: {}\nNext step: {}\nDone criteria: {}",
            focus.status.trim(),
            if focus.blocker.trim().is_empty() {
                "none"
            } else {
                focus.blocker.trim()
            },
            focus.next_step.trim(),
            focus.done_criteria.trim()
        ),
        900,
    )
}

fn context_phase_policy<'a>(
    policy: &'a ContextOptimizationPolicy,
    active_phase: &str,
) -> Option<&'a crate::contracts::ContextOptimizationPhasePolicy> {
    policy
        .phases
        .iter()
        .find(|phase| phase.phase.eq_ignore_ascii_case(active_phase))
}

fn build_context_preparation_state(
    policy: &ContextOptimizationPolicy,
    active_phase: &str,
    checkpoints: &[TaskCheckpointRecord],
    artifact: Option<&ContextPreparedArtifact>,
) -> Option<ContextPreparationState> {
    let artifact = artifact?;
    let previous_artifact = previous_prepared_context_artifact(checkpoints);
    let phase_prefix = context_phase_checkpoint_prefix(active_phase);
    let phase_limit_exceeded = if phase_prefix.is_empty() {
        false
    } else {
        let count = count_checkpoints_with_prefix(checkpoints, phase_prefix);
        policy
            .phases
            .iter()
            .find(|phase| phase.phase == active_phase)
            .map(|phase| count >= phase.max_loops)
            .unwrap_or(false)
    };
    Some(ContextPreparationState {
        query_rounds_completed: count_checkpoints_with_prefix(checkpoints, "context_query"),
        rewrite_rounds_completed: count_checkpoints_with_prefix(checkpoints, "context_rewrite"),
        review_rounds_completed: count_checkpoints_with_prefix(
            checkpoints,
            "context_rewrite_review",
        ),
        latest_review_decision: artifact.review.decision.clone(),
        latest_review_note: trim_chars(&artifact.review.note, 280),
        immediate_next_step: trim_chars(&artifact.immediate_next_step, 320),
        missing_evidence: artifact
            .review
            .missing_evidence
            .iter()
            .take(6)
            .map(|item| trim_chars(item, 160))
            .collect(),
        weak_blocks: artifact
            .review
            .weak_blocks
            .iter()
            .take(6)
            .map(|item| trim_chars(item, 120))
            .collect(),
        repeated_revision_detected: artifacts_repeat_same_blocks(
            artifact,
            previous_artifact.as_ref(),
        ),
        phase_limit_exceeded,
    })
}

fn build_context_optimization_summary(
    policy: &ContextOptimizationPolicy,
    active_phase: String,
) -> ContextOptimizationSummary {
    let phase_policy = context_phase_policy(policy, &active_phase);
    ContextOptimizationSummary {
        max_questions: policy.max_questions,
        max_matches_per_question: policy.max_matches_per_question,
        max_match_chars: policy.max_match_chars,
        max_total_loops: policy.max_total_loops,
        required_block_ids: policy.required_block_ids.clone(),
        block_budgets: policy
            .blocks
            .iter()
            .map(|block| ContextOptimizationBlockBudget {
                block_id: block.block_id.clone(),
                title: block.title.clone(),
                token_budget: block.token_budget,
                required: block.required,
                goal: block.goal.clone(),
            })
            .collect(),
        surfaces: policy.surfaces.clone(),
        negative_signals: policy.negative_signals.clone(),
        positive_signals: policy.positive_signals.clone(),
        assessment_dimensions: policy.assessment_dimensions.clone(),
        active_phase,
        phase_goal: phase_policy
            .map(|phase| phase.goal.clone())
            .unwrap_or_default(),
        phase_required_outputs: phase_policy
            .map(|phase| phase.required_outputs.clone())
            .unwrap_or_default(),
        phase_allowed_review_decisions: phase_policy
            .map(|phase| phase.allowed_review_decisions.clone())
            .unwrap_or_default(),
        go_rule: policy.go_rule.clone(),
        note_formula: policy.note_formula.clone(),
        note_bands: policy.note_bands.clone(),
        note_guardrails: policy.note_guardrails.clone(),
    }
}

fn build_context_preparation_contract(
    policy: &ContextOptimizationPolicy,
    active_phase: &str,
) -> ContextPreparationContract {
    let phase_policy = context_phase_policy(policy, active_phase);
    ContextPreparationContract {
        active_phase: active_phase.to_string(),
        phase_goal: phase_policy
            .map(|phase| phase.goal.clone())
            .unwrap_or_default(),
        total_max_loops: policy.max_total_loops.max(1),
        phase_max_loops: phase_policy
            .map(|phase| phase.max_loops.max(1))
            .unwrap_or(4),
        required_outputs: phase_policy
            .map(|phase| phase.required_outputs.clone())
            .unwrap_or_default(),
        allowed_review_decisions: phase_policy
            .map(|phase| phase.allowed_review_decisions.clone())
            .unwrap_or_default(),
        required_block_ids: policy.required_block_ids.clone(),
        block_budgets: policy
            .blocks
            .iter()
            .map(|block| ContextOptimizationBlockBudget {
                block_id: block.block_id.clone(),
                title: block.title.clone(),
                token_budget: block.token_budget,
                required: block.required,
                goal: block.goal.clone(),
            })
            .collect(),
        go_rule: policy.go_rule.clone(),
    }
}

fn build_context_preparation_review_contract(
    policy: &ContextOptimizationPolicy,
    active_phase: &str,
) -> ContextPreparationReviewContract {
    let (surface_limit, negative_limit, positive_limit, dimension_limit) =
        if active_phase.eq_ignore_ascii_case("query_plan") {
            (4, 6, 3, 3)
        } else if active_phase.eq_ignore_ascii_case("review") {
            (6, 10, 5, 5)
        } else {
            (5, 8, 4, 4)
        };

    ContextPreparationReviewContract {
        surfaces: policy
            .surfaces
            .iter()
            .take(surface_limit)
            .map(|surface| ContextPreparationSurfaceCue {
                surface_id: surface.surface_id.clone(),
                title: surface.title.clone(),
                goal: trim_chars(&surface.goal, 140),
            })
            .collect(),
        negative_signals: policy
            .negative_signals
            .iter()
            .take(negative_limit)
            .map(|signal| ContextPreparationSignalCue {
                signal_id: signal.signal_id.clone(),
                title: signal.title.clone(),
                polarity: signal.polarity.clone(),
                points: signal.points,
                surface_id: signal.surface_id.clone(),
                note: trim_chars(&signal.review_signal, 160),
            })
            .collect(),
        positive_signals: policy
            .positive_signals
            .iter()
            .take(positive_limit)
            .map(|signal| ContextPreparationSignalCue {
                signal_id: signal.signal_id.clone(),
                title: signal.title.clone(),
                polarity: signal.polarity.clone(),
                points: signal.points,
                surface_id: signal.surface_id.clone(),
                note: trim_chars(&signal.criterion, 160),
            })
            .collect(),
        assessment_dimensions: policy
            .assessment_dimensions
            .iter()
            .take(dimension_limit)
            .map(|dimension| ContextPreparationAssessmentCue {
                dimension_id: dimension.dimension_id.clone(),
                title: dimension.title.clone(),
                weight: dimension.weight,
                goal: trim_chars(&dimension.goal, 140),
            })
            .collect(),
        note_guardrails: policy
            .note_guardrails
            .iter()
            .take(4)
            .map(|item| trim_chars(item, 120))
            .collect(),
    }
}

fn is_preparation_context_mode(mode: &str) -> bool {
    matches!(
        mode,
        "preparation" | "preparation_query" | "preparation_rewrite" | "preparation_review"
    )
}

fn keep_preparation_raw_inclusion(source_kind: &str) -> bool {
    matches!(
        source_kind,
        "task_detail"
            | "task_checkpoint"
            | "parent_task"
            | "parent_task_checkpoint"
            | "installation_bootstrap"
            | "definition_of_done_policy"
            | "installation_tool_smoke"
            | "owner_operation_skill"
            | "owner_operation_contract"
            | "recent_owner_outcome"
    )
}

fn compact_context_query_answers_for_preparation(
    answers: &mut Vec<ContextQueryAnswer>,
    limit: usize,
) {
    answers.truncate(limit);
    for answer in answers.iter_mut() {
        answer.question = trim_chars(&answer.question, 220);
        answer.why = trim_chars(&answer.why, 180);
        answer.matches.truncate(3);
        for hit in answer.matches.iter_mut() {
            hit.summary = trim_chars(&hit.summary, 140);
            hit.excerpt = trim_chars(&hit.excerpt, 220);
        }
    }
}

fn compact_context_distillation(
    distillation: &mut ContextDistilledArtifact,
    narrative_limit: usize,
    anchor_limit: usize,
    system_anchor_limit: usize,
    ref_limit: usize,
) {
    distillation.continuity_narrative =
        trim_chars(&distillation.continuity_narrative, narrative_limit);
    distillation.continuity_artifacts.truncate(anchor_limit.max(4));
    for artifact in distillation.continuity_artifacts.iter_mut() {
        artifact.label = trim_chars(&artifact.label, 70);
        artifact.summary = trim_chars(&artifact.summary, 180);
        artifact.source_ref = trim_chars(&artifact.source_ref, 140);
    }
    distillation.continuity_anchors.truncate(anchor_limit);
    for anchor in distillation.continuity_anchors.iter_mut() {
        anchor.title = trim_chars(&anchor.title, 70);
        anchor.content = trim_chars(&anchor.content, 220);
        anchor.evidence_refs.truncate(5);
    }
    distillation
        .system_continuity_anchors
        .truncate(system_anchor_limit);
    for anchor in distillation.system_continuity_anchors.iter_mut() {
        anchor.title = trim_chars(&anchor.title, 70);
        anchor.content = trim_chars(&anchor.content, 180);
        anchor.evidence_refs.truncate(4);
    }
    distillation.active_focus.status = trim_chars(&distillation.active_focus.status, 180);
    distillation.active_focus.blocker = trim_chars(&distillation.active_focus.blocker, 180);
    distillation.active_focus.next_step = trim_chars(&distillation.active_focus.next_step, 180);
    distillation.active_focus.done_criteria =
        trim_chars(&distillation.active_focus.done_criteria, 180);
    distillation.active_focus.evidence_refs.truncate(6);
    if let Some(snapshot) = distillation.snapshot.as_mut() {
        snapshot.summary = trim_chars(&snapshot.summary, 180);
        snapshot.bullet_points = snapshot
            .bullet_points
            .iter()
            .take(5)
            .map(|item| trim_chars(item, 120))
            .collect();
        snapshot.evidence_refs.truncate(4);
    }
    distillation.historical_retrieval_refs.truncate(ref_limit);
    for item in distillation.historical_retrieval_refs.iter_mut() {
        item.label = trim_chars(&item.label, 90);
    }
}

fn compact_context_package_for_preparation(package: &mut ContextPackage) {
    if !is_preparation_context_mode(&package.context_mode) {
        return;
    }
    let active_phase = package.context_optimization.active_phase.as_str();
    let query_phase = active_phase.eq_ignore_ascii_case("query_plan");
    let review_phase = active_phase.eq_ignore_ascii_case("review");

    package.rationale = trim_chars(&package.rationale, 220);
    package.preferred_operating_goal = trim_chars(&package.preferred_operating_goal, 120);
    package.loop_safety.principle = trim_chars(&package.loop_safety.principle, 160);
    package.loop_safety.priority_law = trim_chars(&package.loop_safety.priority_law, 160);
    package.loop_safety.known_failure_modes = package
        .loop_safety
        .known_failure_modes
        .iter()
        .take(3)
        .map(|item| trim_chars(item, 110))
        .collect();
    package.execution_authority.principle = trim_chars(&package.execution_authority.principle, 140);
    package.execution_authority.escalation_rule =
        trim_chars(&package.execution_authority.escalation_rule, 140);
    package.brain_access.current_runtime_kleinhirn =
        trim_chars(&package.brain_access.current_runtime_kleinhirn, 90);
    package.brain_access.local_selected_kleinhirn =
        trim_chars(&package.brain_access.local_selected_kleinhirn, 90);
    package.brain_access.browser_vision_selected_kleinhirn =
        trim_chars(&package.brain_access.browser_vision_selected_kleinhirn, 90);
    package.brain_access.local_kleinhirn_candidates.truncate(1);
    package.brain_access.grosshirn_candidates.truncate(2);
    package.brain_access.operating_rule = trim_chars(&package.brain_access.operating_rule, 160);
    package.skill_system.sync_rule = trim_chars(&package.skill_system.sync_rule, 140);
    package.skill_system.operating_rule = trim_chars(&package.skill_system.operating_rule, 140);
    package.context_governance.principle = trim_chars(&package.context_governance.principle, 140);
    package.context_governance.emergency_boundary =
        trim_chars(&package.context_governance.emergency_boundary, 140);
    package.context_governance.normal_actions = package
        .context_governance
        .normal_actions
        .iter()
        .take(4)
        .map(|item| trim_chars(item, 110))
        .collect();
    package.self_preservation_stage.notes = trim_chars(&package.self_preservation_stage.notes, 100);
    package.host_survival.operating_rule = trim_chars(&package.host_survival.operating_rule, 140);
    package.focus_state.note = trim_chars(&package.focus_state.note, 120);
    package.owner_calibration_summary = package
        .owner_calibration_summary
        .as_ref()
        .map(|item| trim_chars(item, 180))
        .filter(|item| !item.is_empty() && query_phase);
    package.learning_summaries.working_set = None;
    package.learning_summaries.operational = package
        .learning_summaries
        .operational
        .as_ref()
        .map(|item| trim_chars(item, 120))
        .filter(|_| query_phase);
    package.learning_summaries.general = None;
    package.learning_summaries.negative = package
        .learning_summaries
        .negative
        .as_ref()
        .map(|item| trim_chars(item, 120))
        .filter(|_| query_phase || review_phase);
    package.people_working_set = None;
    package.recent_boot_entries.clear();
    package.recent_bios_dialogue.clear();
    package
        .relevant_memory_items
        .truncate(if query_phase { 1 } else { 1 });
    for item in package.relevant_memory_items.iter_mut() {
        item.summary = trim_chars(&item.summary, 120);
    }
    package
        .relevant_learning_entries
        .truncate(if review_phase { 2 } else { 1 });
    for item in package.relevant_learning_entries.iter_mut() {
        item.summary = trim_chars(&item.summary, 120);
        item.applicability = trim_chars(&item.applicability, 120);
    }
    package.relevant_people.clear();
    package.pending_proactive_contacts.clear();
    package
        .recent_task_checkpoints
        .truncate(if query_phase { 2 } else { 3 });
    for item in package.recent_task_checkpoints.iter_mut() {
        item.summary = trim_chars(&item.summary, 160);
        item.detail = trim_chars(&item.detail, 260);
    }
    package
        .recent_turn_signals
        .truncate(if query_phase { 0 } else { 1 });
    for item in package.recent_turn_signals.iter_mut() {
        item.message = trim_chars(&item.message, 140);
    }
    package
        .exec_sessions
        .truncate(if query_phase { 0 } else { 1 });
    for item in package.exec_sessions.iter_mut() {
        item.stdout = trim_chars(&item.stdout, 220);
        item.stderr = trim_chars(&item.stderr, 220);
        item.command.truncate(6);
    }
    package
        .available_skills
        .truncate(if query_phase { 1 } else { 4 });
    for item in package.available_skills.iter_mut() {
        item.description = trim_chars(&item.description, 100);
    }
    package
        .raw_inclusions
        .retain(|item| keep_preparation_raw_inclusion(&item.source_kind));
    package
        .raw_inclusions
        .truncate(if query_phase { 5 } else { 6 });
    for item in package.raw_inclusions.iter_mut() {
        item.content = trim_chars(&item.content, 420);
    }
    package.context_optimization.surfaces = package
        .context_optimization
        .surfaces
        .iter()
        .take(0)
        .cloned()
        .collect();
    package.context_optimization.negative_signals.clear();
    package.context_optimization.positive_signals.clear();
    package.context_optimization.assessment_dimensions.clear();
    package.context_optimization.block_budgets.clear();
    package.context_optimization.note_guardrails.clear();
    package.context_optimization.note_bands.clear();
    package.context_optimization.phase_required_outputs = package
        .context_optimization
        .phase_required_outputs
        .iter()
        .take(if query_phase { 4 } else { 6 })
        .cloned()
        .collect();
    package.context_optimization.phase_allowed_review_decisions = package
        .context_optimization
        .phase_allowed_review_decisions
        .iter()
        .take(3)
        .cloned()
        .collect();
    package.context_optimization.required_block_ids.clear();
    package.context_optimization.phase_goal.clear();
    package.context_optimization.go_rule.clear();
    package.context_optimization.note_formula.clear();
    package.context_query_contract.objective =
        trim_chars(&package.context_query_contract.objective, 140);
    package
        .context_query_contract
        .allowed_source_kinds
        .truncate(if query_phase { 10 } else { 14 });
    package
        .context_query_contract
        .allowed_query_modes
        .truncate(if query_phase { 2 } else { 4 });
    package
        .context_query_contract
        .required_question_fields
        .truncate(if query_phase { 4 } else { 6 });
    package.context_query_contract.provenance_rule =
        trim_chars(&package.context_query_contract.provenance_rule, 140);
    package.context_query_contract.embedding_search_note =
        trim_chars(&package.context_query_contract.embedding_search_note, 120);
    package.task_brief.detail = trim_chars(
        &package.task_brief.detail,
        if query_phase { 1400 } else { 2200 },
    );
    if let Some(distillation) = package.context_distillation.as_mut() {
        compact_context_distillation(
            distillation,
            if query_phase { 420 } else { 620 },
            if review_phase { 6 } else { 4 },
            if review_phase { 4 } else { 3 },
            if review_phase { 12 } else { 8 },
        );
    }
    compact_context_query_answers_for_preparation(
        &mut package.context_query_answers,
        if review_phase { 4 } else { 6 },
    );
    package.retrieval_notes = vec![
        "This package is only for context optimization, not for solving the parent task directly.".to_string(),
        "Rewrite the execution handoff from `contextQueryAnswers` and verified raw inclusions only.".to_string(),
        "If `contextDistillation` exists, preserve its workstream continuity and hard anchors while still revising only the weak parts.".to_string(),
        "Keep tempting but irrelevant owner, BIOS, and broad repo history out unless it changes the next step.".to_string(),
        "Do not exceed `contextOptimization.maxTotalLoops`; if the package is still weak, say so in review instead of retrying blindly.".to_string(),
        "Shorten optional prose before dropping required machine-readable fields.".to_string(),
    ];
}

fn compact_context_package_for_execution_handoff(package: &mut ContextPackage) {
    if is_preparation_context_mode(&package.context_mode) {
        return;
    }
    if package.prepared_context_artifact.is_none() && package.context_distillation.is_none() {
        return;
    }

    package.rationale = trim_chars(&package.rationale, 220);
    package.preferred_operating_goal = trim_chars(&package.preferred_operating_goal, 120);
    package.owner_calibration_summary = None;
    package.recent_boot_entries.clear();
    package.recent_bios_dialogue.clear();
    package.relevant_memory_items.truncate(1);
    for item in package.relevant_memory_items.iter_mut() {
        item.summary = trim_chars(&item.summary, 120);
    }
    package.relevant_learning_entries.truncate(1);
    for item in package.relevant_learning_entries.iter_mut() {
        item.summary = trim_chars(&item.summary, 120);
        item.applicability = trim_chars(&item.applicability, 120);
    }
    package.relevant_people.clear();
    package.pending_proactive_contacts.clear();
    package.recent_task_checkpoints.truncate(2);
    for item in package.recent_task_checkpoints.iter_mut() {
        item.summary = trim_chars(&item.summary, 140);
        item.detail = trim_chars(&item.detail, 200);
    }
    package.recent_turn_signals.truncate(1);
    for item in package.recent_turn_signals.iter_mut() {
        item.message = trim_chars(&item.message, 120);
    }
    package.exec_sessions.truncate(1);
    for item in package.exec_sessions.iter_mut() {
        item.stdout = trim_chars(&item.stdout, 180);
        item.stderr = trim_chars(&item.stderr, 180);
        item.command.truncate(6);
    }
    package.available_skills.truncate(4);
    for item in package.available_skills.iter_mut() {
        item.description = trim_chars(&item.description, 90);
    }
    package.raw_inclusions.retain(|item| {
        matches!(
            item.source_kind.as_str(),
            "task_detail"
                | "task_checkpoint"
                | "parent_task"
                | "parent_task_checkpoint"
                | "definition_of_done_policy"
                | "repo_operation_skill"
                | "system_capability_contract"
                | "recent_task_outcome"
                | "current_task_machine_evidence"
        )
    });
    package
        .raw_inclusions
        .sort_by_key(|item| match item.source_kind.as_str() {
            "current_task_machine_evidence" => 0,
            "system_capability_contract" => 1,
            "repo_operation_skill" => 2,
            "task_definition_of_done_policy" => 3,
            "task_detail" => 4,
            "task_checkpoint" => 5,
            "parent_task" => 6,
            "parent_task_checkpoint" => 7,
            "recent_task_outcome" => 8,
            _ => 9,
        });
    package.raw_inclusions.truncate(9);
    for item in package.raw_inclusions.iter_mut() {
        item.content = trim_chars(
            &item.content,
            if item.source_kind == "current_task_machine_evidence" {
                520
            } else {
                320
            },
        );
    }
    package.context_query_answers.clear();
    if let Some(distillation) = package.context_distillation.as_mut() {
        compact_context_distillation(distillation, 700, 6, 4, 12);
    }
    if let Some(artifact) = package.prepared_context_artifact.as_mut() {
        artifact.questions.clear();
        artifact.blocks.retain(|block| {
            matches!(
                block.block_id.as_str(),
                "goal_and_authority"
                    | "definition_of_done"
                    | "verified_world_state"
                    | "relevant_artifacts"
                    | "next_action_only"
            )
        });
        for block in artifact.blocks.iter_mut() {
            block.title = trim_chars(&block.title, 70);
            block.content = trim_chars(&block.content, 180);
            block.why_included = trim_chars(&block.why_included, 120);
            block.evidence_refs.truncate(5);
            block.omitted_items.truncate(4);
        }
    }
    package.preparation_state = None;
    package.preparation_contract = None;
    package.preparation_review_contract = None;
    package.context_optimization.required_block_ids.clear();
    package.context_optimization.block_budgets.clear();
    package.context_optimization.surfaces.clear();
    package.context_optimization.negative_signals.clear();
    package.context_optimization.positive_signals.clear();
    package.context_optimization.assessment_dimensions.clear();
    package.context_optimization.phase_goal.clear();
    package.context_optimization.go_rule.clear();
    package.context_optimization.note_formula.clear();
    package.context_optimization.note_bands.clear();
    package.context_optimization.note_guardrails.clear();
    package.retrieval_notes = vec![
        "Prepared context handoff is active. Use `contextDistillation.activeFocus` as the primary execution brief for this run.".to_string(),
        "Use `contextDistillation.continuityNarrative` plus continuity anchors to preserve multi-step continuity, and keep `systemContinuityAnchors` for reprioritization or reflection.".to_string(),
        "Use `contextDistillation.continuityArtifacts` for the concrete files, contracts, sessions, and checkpoints that must survive the next turn.".to_string(),
        "Treat `preparedContextArtifact.blocks` as provenance behind the distillation, not as license to reopen broad raw-history scavenging.".to_string(),
        "If exact historical detail is needed, prefer `contextDistillation.historicalRetrievalRefs` and narrow SQLite/embedding reload over whole-history expansion.".to_string(),
    ];
}

fn build_context_query_contract_summary(
    contract: &ContextQueryToolContract,
    embedding_target: Option<&ContextEmbeddingTarget>,
) -> ContextQueryContractSummary {
    let embedding_search_available =
        contract.embedding_search_available && embedding_target.is_some();
    let embedding_search_note = if embedding_search_available {
        let target = embedding_target.expect("embedding target should exist when available");
        format!(
            "Embedding retrieval is configured through {} with model {}.",
            target.base_url, target.model_id
        )
    } else if contract.embedding_search_available {
        "Embedding retrieval is supported by this repo, but no context-embedding runtime is configured yet.".to_string()
    } else {
        contract.embedding_search_note.clone()
    };
    ContextQueryContractSummary {
        objective: contract.objective.clone(),
        allowed_source_kinds: contract.allowed_source_kinds.clone(),
        allowed_query_modes: contract.allowed_query_modes.clone(),
        default_query_mode: contract.default_query_mode.clone(),
        required_question_fields: contract.required_question_fields.clone(),
        max_questions: contract.max_questions,
        provenance_rule: contract.provenance_rule.clone(),
        embedding_search_available,
        embedding_search_note,
    }
}

fn parse_runtime_context_env(paths: &Paths) -> BTreeMap<String, String> {
    let mut env_map = BTreeMap::new();
    for (key, value) in env::vars() {
        if key.starts_with("CTO_AGENT_CONTEXT_EMBEDDING_")
            || key == "CTO_AGENT_KLEINHIRN_BASE_URL"
            || key == "CTO_AGENT_KLEINHIRN_API_KEY"
        {
            env_map.insert(key, value);
        }
    }
    let env_path = paths.root.join("runtime/kleinhirn.env");
    if let Ok(text) = fs::read_to_string(env_path) {
        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let Some((key, raw_value)) = trimmed.split_once('=') else {
                continue;
            };
            env_map
                .entry(key.trim().to_string())
                .or_insert_with(|| unquote_runtime_context_env_value(raw_value.trim()));
        }
    }
    env_map
}

fn unquote_runtime_context_env_value(value: &str) -> String {
    let bytes = value.as_bytes();
    if bytes.len() >= 2
        && ((bytes[0] == b'\'' && bytes[bytes.len() - 1] == b'\'')
            || (bytes[0] == b'"' && bytes[bytes.len() - 1] == b'"'))
    {
        value[1..value.len() - 1].replace("'\\''", "'")
    } else {
        value.to_string()
    }
}

fn context_env_truthy(value: Option<&String>, default: bool) -> bool {
    match value.map(|value| value.trim().to_ascii_lowercase()) {
        Some(value) if matches!(value.as_str(), "0" | "false" | "no" | "off") => false,
        Some(value) if matches!(value.as_str(), "1" | "true" | "yes" | "on") => true,
        Some(_) => true,
        None => default,
    }
}

fn resolve_context_embedding_target(paths: &Paths) -> Option<ContextEmbeddingTarget> {
    let env_map = parse_runtime_context_env(paths);
    let explicitly_configured = env_map
        .keys()
        .any(|key| key.starts_with("CTO_AGENT_CONTEXT_EMBEDDING_"));
    if !explicitly_configured {
        return None;
    }
    if !context_env_truthy(env_map.get("CTO_AGENT_CONTEXT_EMBEDDING_ENABLED"), true) {
        return None;
    }
    let model_id = env_map
        .get("CTO_AGENT_CONTEXT_EMBEDDING_RUNTIME_MODEL")
        .cloned()
        .or_else(|| env_map.get("CTO_AGENT_CONTEXT_EMBEDDING_MODEL").cloned())
        .unwrap_or_else(|| "Qwen/Qwen3-Embedding-0.6B".to_string());
    let api_key = env_map
        .get("CTO_AGENT_CONTEXT_EMBEDDING_API_KEY")
        .cloned()
        .or_else(|| env_map.get("CTO_AGENT_KLEINHIRN_API_KEY").cloned())
        .unwrap_or_else(|| "local-context-embedding".to_string());
    let base_url = env_map
        .get("CTO_AGENT_CONTEXT_EMBEDDING_BASE_URL")
        .cloned()
        .or_else(|| {
            env_map
                .get("CTO_AGENT_CONTEXT_EMBEDDING_PORT")
                .map(|port| format!("http://127.0.0.1:{port}/v1"))
        })
        .unwrap_or_else(|| "http://127.0.0.1:1235/v1".to_string());
    Some(ContextEmbeddingTarget {
        base_url,
        api_key,
        model_id,
        chunk_chars: env_map
            .get("CTO_AGENT_CONTEXT_EMBEDDING_CHUNK_CHARS")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(900),
        chunk_overlap_chars: env_map
            .get("CTO_AGENT_CONTEXT_EMBEDDING_CHUNK_OVERLAP")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(120),
        max_batch_size: env_map
            .get("CTO_AGENT_CONTEXT_EMBEDDING_MAX_BATCH_SIZE")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(12)
            .max(1),
        document_instruction: env_map
            .get("CTO_AGENT_CONTEXT_EMBEDDING_DOCUMENT_INSTRUCTION")
            .cloned()
            .unwrap_or_else(|| {
                "Represent this CTO-Agent context artifact for retrieval.".to_string()
            }),
        query_instruction: env_map
            .get("CTO_AGENT_CONTEXT_EMBEDDING_QUERY_INSTRUCTION")
            .cloned()
            .unwrap_or_else(|| {
                "Represent this CTO-Agent context retrieval question for semantic search."
                    .to_string()
            }),
    })
}

fn append_context_distillation_candidates_from_package(
    candidates: &mut Vec<ContextQueryCandidate>,
    task_id: i64,
    task_title: &str,
    package_json: &str,
) {
    let Ok(value) = serde_json::from_str::<serde_json::Value>(package_json) else {
        return;
    };
    let Some(distillation_value) = value.get("contextDistillation").cloned() else {
        return;
    };
    let Ok(distillation) = serde_json::from_value::<ContextDistilledArtifact>(distillation_value)
    else {
        return;
    };

    if !distillation.continuity_narrative.trim().is_empty() {
        candidates.push(ContextQueryCandidate {
            source_kind: "context_distillation_story".to_string(),
            source_ref: format!("distillation:story:{task_id}"),
            summary: format!("Continuity narrative for task #{task_id} {task_title}"),
            text: distillation.continuity_narrative,
        });
    }
    for artifact in distillation.continuity_artifacts {
        candidates.push(ContextQueryCandidate {
            source_kind: "context_distillation_artifact".to_string(),
            source_ref: format!("distillation:artifact:{task_id}:{}", artifact.artifact_id),
            summary: format!("Continuity artifact {} for task #{task_id}", artifact.label),
            text: format!(
                "{}\nKind: {}\nSource: {}\n\n{}",
                artifact.label, artifact.kind, artifact.source_ref, artifact.summary
            ),
        });
    }
    for anchor in distillation.continuity_anchors {
        candidates.push(ContextQueryCandidate {
            source_kind: "context_distillation_anchor".to_string(),
            source_ref: format!("distillation:anchor:{task_id}:{}", anchor.anchor_id),
            summary: format!("Continuity anchor {} for task #{task_id}", anchor.title),
            text: format!("{}\n\n{}", anchor.title, anchor.content),
        });
    }
    for anchor in distillation.system_continuity_anchors {
        candidates.push(ContextQueryCandidate {
            source_kind: "context_distillation_system_anchor".to_string(),
            source_ref: format!("distillation:system:{task_id}:{}", anchor.anchor_id),
            summary: format!(
                "System continuity anchor {} for task #{task_id}",
                anchor.title
            ),
            text: format!("{}\n\n{}", anchor.title, anchor.content),
        });
    }
    candidates.push(ContextQueryCandidate {
        source_kind: "context_distillation_focus".to_string(),
        source_ref: format!("distillation:focus:{task_id}"),
        summary: format!("Active focus for task #{task_id} {task_title}"),
        text: format!(
            "Status: {}\nBlocker: {}\nNext: {}\nDone: {}",
            distillation.active_focus.status,
            distillation.active_focus.blocker,
            distillation.active_focus.next_step,
            distillation.active_focus.done_criteria
        ),
    });
    if let Some(snapshot) = distillation.snapshot {
        candidates.push(ContextQueryCandidate {
            source_kind: "context_distillation_snapshot".to_string(),
            source_ref: format!("distillation:snapshot:{task_id}"),
            summary: format!("Context snapshot for task #{task_id} {task_title}"),
            text: format!(
                "{}\n\n{}",
                snapshot.summary,
                snapshot.bullet_points.join("\n")
            ),
        });
    }
}

fn build_context_query_candidates(paths: &Paths, task: &TaskRecord) -> Vec<ContextQueryCandidate> {
    let mut candidates = Vec::new();
    let mut task_text = format!("{}\n\n{}", task.title, task.detail);
    if let Some(output) = task.last_output.as_deref()
        && !output.trim().is_empty()
    {
        task_text.push_str("\n\nLatest task output:\n");
        task_text.push_str(output.trim());
    }
    for checkpoint in list_task_checkpoints(paths, task.id, 4).unwrap_or_default() {
        task_text.push_str("\n\nCurrent task checkpoint:\n");
        task_text.push_str(&checkpoint.summary);
        task_text.push_str("\n\n");
        task_text.push_str(&checkpoint.detail);
        candidates.push(ContextQueryCandidate {
            source_kind: "task_checkpoint".to_string(),
            source_ref: format!("task:{}:{}", task.id, checkpoint.checkpoint_kind),
            summary: format!(
                "Current checkpoint {}: {}",
                checkpoint.checkpoint_kind, checkpoint.summary
            ),
            text: format!("{}\n\n{}", checkpoint.summary, checkpoint.detail),
        });
    }
    candidates.push(ContextQueryCandidate {
        source_kind: "task_detail".to_string(),
        source_ref: format!("task:{}", task.id),
        summary: format!("Task #{} {} ({})", task.id, task.title, task.task_kind),
        text: task_text,
    });
    if let Ok(Some(package)) = latest_context_package_for_task(paths, task.id) {
        append_context_distillation_candidates_from_package(
            &mut candidates,
            task.id,
            &task.title,
            &package.package_json,
        );
    }
    if let Some(parent_task_id) = task.parent_task_id
        && let Ok(Some(parent_task)) = load_task_by_id(paths, parent_task_id)
    {
        candidates.push(ContextQueryCandidate {
            source_kind: "parent_task".to_string(),
            source_ref: format!("task:{}", parent_task.id),
            summary: format!("Parent task #{} {}", parent_task.id, parent_task.title),
            text: format!("{}\n\n{}", parent_task.title, parent_task.detail),
        });
        for checkpoint in list_task_checkpoints(paths, parent_task_id, 4).unwrap_or_default() {
            candidates.push(ContextQueryCandidate {
                source_kind: "parent_task_checkpoint".to_string(),
                source_ref: format!("task:{}:{}", parent_task_id, checkpoint.checkpoint_kind),
                summary: format!(
                    "Parent checkpoint {}: {}",
                    checkpoint.checkpoint_kind, checkpoint.summary
                ),
                text: format!("{}\n\n{}", checkpoint.summary, checkpoint.detail),
            });
        }
        if let Ok(Some(package)) = latest_context_package_for_task(paths, parent_task_id) {
            append_context_distillation_candidates_from_package(
                &mut candidates,
                parent_task_id,
                &parent_task.title,
                &package.package_json,
            );
        }
    }
    for outcome in list_recent_task_outcomes(paths, task.id, 8).unwrap_or_default() {
        let mut text = String::new();
        text.push_str(&outcome.title);
        text.push_str("\n\n");
        text.push_str(&outcome.detail);
        if let Some(summary) = outcome.last_checkpoint_summary.as_deref() {
            if !summary.trim().is_empty() {
                text.push_str("\n\nCheckpoint:\n");
                text.push_str(summary.trim());
            }
        }
        if let Some(output) = outcome.last_output.as_deref()
            && !output.trim().is_empty()
        {
            text.push_str("\n\nOutput:\n");
            text.push_str(output.trim());
        }
        candidates.push(ContextQueryCandidate {
            source_kind: "recent_task_outcome".to_string(),
            source_ref: format!("task:{}", outcome.id),
            summary: format!("Recent task outcome #{} {}", outcome.id, outcome.title),
            text,
        });
        if let Ok(Some(package)) = latest_context_package_for_task(paths, outcome.id) {
            append_context_distillation_candidates_from_package(
                &mut candidates,
                outcome.id,
                &outcome.title,
                &package.package_json,
            );
        }
    }
    for item in list_memory_items(paths, 18).unwrap_or_default() {
        candidates.push(ContextQueryCandidate {
            source_kind: "memory_item".to_string(),
            source_ref: format!("memory:{}:{}", item.kind, item.created_at),
            summary: format!("Memory {} {}", item.kind, item.summary),
            text: format!("{}\n\n{}", item.summary, item.detail),
        });
    }
    for entry in list_active_learning_entries(paths, 18).unwrap_or_default() {
        candidates.push(ContextQueryCandidate {
            source_kind: "learning_entry".to_string(),
            source_ref: format!("learning:{}", entry.id),
            summary: format!("Learning {} {}", entry.learning_class, entry.summary),
            text: format!(
                "{}\n\n{}\n\nApplicability: {}",
                entry.summary, entry.detail, entry.applicability
            ),
        });
    }
    for person in list_person_profiles(paths, 10).unwrap_or_default() {
        candidates.push(ContextQueryCandidate {
            source_kind: "person_profile".to_string(),
            source_ref: format!("person:{}", person.id),
            summary: format!(
                "Person {} {}",
                person.display_name, person.relationship_kind
            ),
            text: format!(
                "{}\n{}\n{}\n{}",
                person.display_name,
                person.primary_email,
                person.conversation_memory_summary,
                person.notebook_summary
            ),
        });
    }
    for skill in sync_skills(paths)
        .or_else(|_| list_skills(paths))
        .unwrap_or_default()
    {
        candidates.push(ContextQueryCandidate {
            source_kind: "skill".to_string(),
            source_ref: skill.path.clone(),
            summary: format!("Skill {} {}", skill.name, skill.status),
            text: format!("{}\n\n{}", skill.name, skill.notes),
        });
    }
    candidates
}

fn open_context_embedding_db(paths: &Paths) -> Option<Connection> {
    let conn = Connection::open(&paths.runtime_db_path).ok()?;
    conn.pragma_update(None, "foreign_keys", "ON").ok()?;
    conn.pragma_update(None, "temp_store", "MEMORY").ok()?;
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS context_embedding_chunks (
            source_kind TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            embedding_model TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            content_hash TEXT NOT NULL,
            text_chunk TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(source_kind, source_ref, embedding_model, chunk_index)
        );
        CREATE INDEX IF NOT EXISTS idx_context_embedding_chunks_source_model
            ON context_embedding_chunks(source_kind, source_ref, embedding_model, updated_at DESC);",
    )
    .ok()?;
    Some(conn)
}

fn hash_context_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn build_document_embedding_text(
    target: &ContextEmbeddingTarget,
    candidate: &ContextQueryCandidate,
    chunk_text: &str,
) -> String {
    format!(
        "Instruction: {}\nSource kind: {}\nSource ref: {}\nSummary: {}\nContent:\n{}",
        target.document_instruction,
        candidate.source_kind,
        candidate.source_ref,
        candidate.summary,
        chunk_text
    )
}

fn build_query_embedding_text(
    target: &ContextEmbeddingTarget,
    question: &ContextQueryQuestion,
    keywords: &HashSet<String>,
) -> String {
    let mut keyword_list = build_query_terms(question, keywords);
    keyword_list.truncate(12);
    format!(
        "Instruction: {}\nQuestion: {}\nReason: {}\nKeywords: {}",
        target.query_instruction,
        question.question,
        question.why,
        keyword_list.join(", ")
    )
}

fn chunk_context_candidate_text(
    candidate: &ContextQueryCandidate,
    target: &ContextEmbeddingTarget,
) -> Vec<String> {
    let combined = format!("{}\n\n{}", candidate.summary, candidate.text);
    let chars = combined.chars().collect::<Vec<_>>();
    if chars.is_empty() {
        return Vec::new();
    }
    let chunk_chars = target.chunk_chars.max(120);
    let overlap = target.chunk_overlap_chars.min(chunk_chars / 2).min(240);
    let mut chunks = Vec::new();
    let mut start = 0_usize;
    while start < chars.len() {
        let end = (start + chunk_chars).min(chars.len());
        let chunk = chars[start..end]
            .iter()
            .collect::<String>()
            .trim()
            .to_string();
        if !chunk.is_empty() {
            chunks.push(chunk);
        }
        if end >= chars.len() {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    chunks
}

fn parse_embedding_response_vectors(value: &serde_json::Value) -> Option<Vec<Vec<f32>>> {
    let data = value.get("data")?.as_array()?;
    let mut vectors = Vec::with_capacity(data.len());
    for item in data {
        let embedding = item.get("embedding")?.as_array()?;
        let vector = embedding
            .iter()
            .filter_map(|value| value.as_f64().map(|entry| entry as f32))
            .collect::<Vec<_>>();
        if vector.is_empty() {
            return None;
        }
        vectors.push(vector);
    }
    Some(vectors)
}

fn request_embedding_vectors(
    target: &ContextEmbeddingTarget,
    inputs: &[String],
) -> Option<Vec<Vec<f32>>> {
    if inputs.is_empty() {
        return Some(Vec::new());
    }
    let client = Client::builder()
        .timeout(Duration::from_secs(90))
        .build()
        .ok()?;
    let response = client
        .post(format!(
            "{}/embeddings",
            target.base_url.trim_end_matches('/')
        ))
        .bearer_auth(&target.api_key)
        .json(&serde_json::json!({
            "model": target.model_id,
            "input": inputs,
            "encoding_format": "float",
            "truncate_sequence": true
        }))
        .send()
        .ok()?
        .error_for_status()
        .ok()?;
    let payload = response.json::<serde_json::Value>().ok()?;
    parse_embedding_response_vectors(&payload)
}

fn load_cached_embedding_chunks(
    conn: &Connection,
    source_kind: &str,
    source_ref: &str,
    embedding_model: &str,
) -> Vec<CachedEmbeddingChunk> {
    let mut stmt = match conn.prepare(
        "SELECT chunk_index, content_hash, text_chunk, embedding_json
         FROM context_embedding_chunks
         WHERE source_kind = ?1 AND source_ref = ?2 AND embedding_model = ?3
         ORDER BY chunk_index ASC",
    ) {
        Ok(stmt) => stmt,
        Err(_) => return Vec::new(),
    };
    let rows = match stmt.query_map(params![source_kind, source_ref, embedding_model], |row| {
        let chunk_index: i64 = row.get(0)?;
        let content_hash: String = row.get(1)?;
        let text_chunk: String = row.get(2)?;
        let embedding_json: String = row.get(3)?;
        Ok((chunk_index, content_hash, text_chunk, embedding_json))
    }) {
        Ok(rows) => rows,
        Err(_) => return Vec::new(),
    };
    rows.filter_map(|row| {
        let Ok((chunk_index, content_hash, text_chunk, embedding_json)) = row else {
            return None;
        };
        let embedding = serde_json::from_str::<Vec<f32>>(&embedding_json).ok()?;
        Some(CachedEmbeddingChunk {
            source_kind: source_kind.to_string(),
            source_ref: source_ref.to_string(),
            chunk_index: chunk_index.max(0) as usize,
            content_hash,
            text_chunk,
            embedding,
        })
    })
    .collect()
}

fn replace_cached_embedding_chunks(
    conn: &mut Connection,
    source_kind: &str,
    source_ref: &str,
    embedding_model: &str,
    chunks: &[CachedEmbeddingChunk],
) -> bool {
    let tx = match conn.transaction() {
        Ok(tx) => tx,
        Err(_) => return false,
    };
    if tx
        .execute(
            "DELETE FROM context_embedding_chunks
             WHERE source_kind = ?1 AND source_ref = ?2 AND embedding_model = ?3",
            params![source_kind, source_ref, embedding_model],
        )
        .is_err()
    {
        return false;
    }
    let now = now_iso();
    for chunk in chunks {
        let embedding_json = match serde_json::to_string(&chunk.embedding) {
            Ok(value) => value,
            Err(_) => return false,
        };
        if tx
            .execute(
                "INSERT INTO context_embedding_chunks(
                    source_kind, source_ref, embedding_model, chunk_index, content_hash, text_chunk, embedding_json, updated_at
                 ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                params![
                    chunk.source_kind,
                    chunk.source_ref,
                    embedding_model,
                    chunk.chunk_index as i64,
                    chunk.content_hash,
                    chunk.text_chunk,
                    embedding_json,
                    now
                ],
            )
            .is_err()
        {
            return false;
        }
    }
    tx.commit().is_ok()
}

fn cosine_similarity(left: &[f32], right: &[f32]) -> Option<f64> {
    if left.is_empty() || right.is_empty() || left.len() != right.len() {
        return None;
    }
    let mut dot = 0.0_f64;
    let mut left_norm = 0.0_f64;
    let mut right_norm = 0.0_f64;
    for (a, b) in left.iter().zip(right.iter()) {
        let a = *a as f64;
        let b = *b as f64;
        dot += a * b;
        left_norm += a * a;
        right_norm += b * b;
    }
    if left_norm <= f64::EPSILON || right_norm <= f64::EPSILON {
        return None;
    }
    Some(dot / (left_norm.sqrt() * right_norm.sqrt()))
}

fn ensure_candidate_embedding_cache(
    paths: &Paths,
    target: &ContextEmbeddingTarget,
    candidates: &[&ContextQueryCandidate],
) -> Option<HashMap<(String, String), Vec<CachedEmbeddingChunk>>> {
    let mut conn = open_context_embedding_db(paths)?;
    let mut cached_by_source = HashMap::new();
    let mut pending_by_source =
        HashMap::<(String, String), Vec<(usize, String, String, String)>>::new();
    let mut pending_keys = Vec::<(String, String)>::new();

    for candidate in candidates {
        let source_key = (candidate.source_kind.clone(), candidate.source_ref.clone());
        let chunks = chunk_context_candidate_text(candidate, target)
            .into_iter()
            .enumerate()
            .map(|(index, chunk_text)| {
                let embedding_text = build_document_embedding_text(target, candidate, &chunk_text);
                (
                    index,
                    hash_context_text(&embedding_text),
                    chunk_text,
                    embedding_text,
                )
            })
            .collect::<Vec<_>>();
        let existing = load_cached_embedding_chunks(
            &conn,
            &candidate.source_kind,
            &candidate.source_ref,
            &target.model_id,
        );
        let cache_matches = existing.len() == chunks.len()
            && existing.iter().zip(chunks.iter()).all(|(cached, current)| {
                cached.chunk_index == current.0 && cached.content_hash == current.1
            });
        if cache_matches {
            cached_by_source.insert(source_key, existing);
        } else {
            pending_by_source.insert(source_key.clone(), chunks);
            pending_keys.push(source_key);
        }
    }

    if pending_by_source.is_empty() {
        return Some(cached_by_source);
    }

    let mut inputs = Vec::new();
    let mut meta = Vec::new();
    for key in &pending_keys {
        if let Some(chunks) = pending_by_source.get(key) {
            for (chunk_index, content_hash, chunk_text, embedding_text) in chunks {
                inputs.push(embedding_text.clone());
                meta.push((
                    key.clone(),
                    *chunk_index,
                    content_hash.clone(),
                    chunk_text.clone(),
                ));
            }
        }
    }

    let mut vectors = Vec::<Vec<f32>>::new();
    for batch in inputs.chunks(target.max_batch_size) {
        let batch_inputs = batch.to_vec();
        let mut batch_vectors = request_embedding_vectors(target, &batch_inputs)?;
        vectors.append(&mut batch_vectors);
    }
    if vectors.len() != meta.len() {
        return None;
    }

    let mut generated = HashMap::<(String, String), Vec<CachedEmbeddingChunk>>::new();
    for (((source_kind, source_ref), chunk_index, content_hash, text_chunk), embedding) in
        meta.into_iter().zip(vectors.into_iter())
    {
        generated
            .entry((source_kind.clone(), source_ref.clone()))
            .or_default()
            .push(CachedEmbeddingChunk {
                source_kind,
                source_ref,
                chunk_index,
                content_hash,
                text_chunk,
                embedding,
            });
    }

    for ((source_kind, source_ref), chunks) in &generated {
        if !replace_cached_embedding_chunks(
            &mut conn,
            source_kind,
            source_ref,
            &target.model_id,
            chunks,
        ) {
            return None;
        }
    }

    cached_by_source.extend(generated);
    Some(cached_by_source)
}

fn build_semantic_query_matches(
    paths: &Paths,
    target: &ContextEmbeddingTarget,
    candidates: &[&ContextQueryCandidate],
    question: &ContextQueryQuestion,
    keywords: &HashSet<String>,
    max_matches: usize,
    max_match_chars: usize,
) -> Option<Vec<ContextQueryMatch>> {
    if candidates.is_empty() {
        return Some(Vec::new());
    }
    let query_vector = request_embedding_vectors(
        target,
        &[build_query_embedding_text(target, question, keywords)],
    )?
    .into_iter()
    .next()?;
    let cached = ensure_candidate_embedding_cache(paths, target, candidates)?;
    let mut scored = Vec::new();
    for candidate in candidates {
        let source_key = (candidate.source_kind.clone(), candidate.source_ref.clone());
        let Some(chunks) = cached.get(&source_key) else {
            continue;
        };
        let best = chunks
            .iter()
            .filter_map(|chunk| {
                cosine_similarity(&query_vector, &chunk.embedding)
                    .map(|similarity| (similarity, chunk))
            })
            .max_by(|left, right| {
                left.0
                    .partial_cmp(&right.0)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        let Some((similarity, best_chunk)) = best else {
            continue;
        };
        if similarity <= 0.0 {
            continue;
        }
        let lexical_bonus = relevance_score(
            &format!("{} {}", candidate.summary, candidate.text),
            keywords,
        );
        let score = ((similarity * 1000.0).round() as usize).saturating_add(lexical_bonus);
        scored.push(ContextQueryMatch {
            source_kind: candidate.source_kind.clone(),
            source_ref: candidate.source_ref.clone(),
            summary: trim_chars(&candidate.summary, 180),
            excerpt: trim_chars(&best_chunk.text_chunk, max_match_chars),
            score,
            query_mode_used: "sqlite_semantic".to_string(),
        });
    }
    scored.sort_by(|left, right| right.score.cmp(&left.score));
    Some(scored.into_iter().take(max_matches).collect())
}

fn filter_query_candidates<'a>(
    candidates: &'a [ContextQueryCandidate],
    source_filters: &[String],
    query_contract: &ContextQueryToolContract,
) -> Vec<&'a ContextQueryCandidate> {
    let allowed_sources = query_contract
        .allowed_source_kinds
        .iter()
        .map(|value| value.to_ascii_lowercase())
        .collect::<HashSet<_>>();
    let effective_filters = source_filters
        .iter()
        .map(|value| value.to_ascii_lowercase())
        .filter(|value| allowed_sources.contains(value))
        .collect::<Vec<_>>();
    candidates
        .iter()
        .filter(|candidate| allowed_sources.contains(&candidate.source_kind.to_ascii_lowercase()))
        .filter(|candidate| {
            effective_filters.is_empty()
                || effective_filters
                    .iter()
                    .any(|source| candidate.source_kind.eq_ignore_ascii_case(source))
        })
        .collect()
}

fn build_ranked_query_hits<'a>(
    candidates: &[&'a ContextQueryCandidate],
    keywords: &HashSet<String>,
) -> Vec<RankedQueryHit<'a>> {
    let mut ranked = candidates
        .iter()
        .filter_map(|candidate| {
            let score = relevance_score(
                &format!("{} {}", candidate.summary, candidate.text),
                keywords,
            );
            if score == 0 {
                return None;
            }
            Some(RankedQueryHit { candidate, score })
        })
        .collect::<Vec<_>>();
    ranked.sort_by(|left, right| right.score.cmp(&left.score));
    ranked
}

fn build_query_terms(question: &ContextQueryQuestion, keywords: &HashSet<String>) -> Vec<String> {
    let mut ordered = Vec::new();
    let mut seen = HashSet::new();
    for term in &question.required_keywords {
        let trimmed = term.trim().to_ascii_lowercase();
        if trimmed.len() >= 2 && seen.insert(trimmed.clone()) {
            ordered.push(trimmed);
        }
    }
    for term in keywords {
        let trimmed = term.trim().to_ascii_lowercase();
        if trimmed.len() >= 2 && seen.insert(trimmed.clone()) {
            ordered.push(trimmed);
        }
    }
    ordered
}

fn build_ranked_query_matches(
    ranked: Vec<RankedQueryHit<'_>>,
    max_matches: usize,
    max_match_chars: usize,
    query_mode_used: &str,
) -> Vec<ContextQueryMatch> {
    ranked
        .into_iter()
        .take(max_matches)
        .map(|hit| ContextQueryMatch {
            source_kind: hit.candidate.source_kind.clone(),
            source_ref: hit.candidate.source_ref.clone(),
            summary: trim_chars(&hit.candidate.summary, 180),
            excerpt: trim_chars(&hit.candidate.text, max_match_chars),
            score: hit.score,
            query_mode_used: query_mode_used.to_string(),
        })
        .collect()
}

fn sqlite_fts_query_string(question: &ContextQueryQuestion, keywords: &HashSet<String>) -> String {
    build_query_terms(question, keywords)
        .into_iter()
        .filter(|term| term.len() >= 2)
        .take(10)
        .map(|term| {
            if term.contains(' ') {
                format!("\"{}\"", term.replace('"', " "))
            } else {
                term
            }
        })
        .collect::<Vec<_>>()
        .join(" OR ")
}

fn build_sqlite_fts_matches(
    candidates: &[&ContextQueryCandidate],
    question: &ContextQueryQuestion,
    keywords: &HashSet<String>,
    max_matches: usize,
    max_match_chars: usize,
) -> Option<Vec<ContextQueryMatch>> {
    if candidates.is_empty() {
        return Some(Vec::new());
    }
    let query = sqlite_fts_query_string(question, keywords);
    if query.trim().is_empty() {
        return Some(Vec::new());
    }
    let conn = Connection::open_in_memory().ok()?;
    conn.execute_batch(
        "CREATE VIRTUAL TABLE context_query_fts \
         USING fts5(source_kind UNINDEXED, source_ref UNINDEXED, summary, body);",
    )
    .ok()?;
    for (index, candidate) in candidates.iter().enumerate() {
        conn.execute(
            "INSERT INTO context_query_fts(rowid, source_kind, source_ref, summary, body) \
             VALUES (?1, ?2, ?3, ?4, ?5)",
            params![
                (index + 1) as i64,
                candidate.source_kind,
                candidate.source_ref,
                candidate.summary,
                candidate.text
            ],
        )
        .ok()?;
    }
    let mut stmt = conn
        .prepare(
            "SELECT rowid, snippet(context_query_fts, 3, '', '', ' … ', 24), bm25(context_query_fts) \
             FROM context_query_fts \
             WHERE context_query_fts MATCH ?1 \
             ORDER BY bm25(context_query_fts) ASC \
             LIMIT ?2",
        )
        .ok()?;
    let rows = stmt
        .query_map(params![query, max_matches as i64], |row| {
            let rowid: i64 = row.get(0)?;
            let excerpt: Option<String> = row.get(1)?;
            let bm25_score: f64 = row.get(2)?;
            Ok((rowid, excerpt.unwrap_or_default(), bm25_score))
        })
        .ok()?;
    let mut matches = Vec::new();
    for (rank_index, row_result) in rows.enumerate() {
        let Ok((rowid, excerpt, bm25_score)) = row_result else {
            continue;
        };
        let candidate_index = rowid.saturating_sub(1) as usize;
        let Some(candidate) = candidates.get(candidate_index) else {
            continue;
        };
        let lexical_score = relevance_score(
            &format!("{} {}", candidate.summary, candidate.text),
            keywords,
        );
        let position_bonus = max_matches.saturating_sub(rank_index).saturating_mul(5);
        let bm25_bonus = (100.0 / (1.0 + bm25_score.abs())).round().max(1.0) as usize;
        matches.push(ContextQueryMatch {
            source_kind: candidate.source_kind.clone(),
            source_ref: candidate.source_ref.clone(),
            summary: trim_chars(&candidate.summary, 180),
            excerpt: if excerpt.trim().is_empty() {
                trim_chars(&candidate.text, max_match_chars)
            } else {
                trim_chars(&excerpt, max_match_chars)
            },
            score: lexical_score + position_bonus + bm25_bonus,
            query_mode_used: "sqlite_fts".to_string(),
        });
    }
    Some(matches)
}

fn build_hybrid_query_matches(
    paths: &Paths,
    embedding_target: Option<&ContextEmbeddingTarget>,
    candidates: &[&ContextQueryCandidate],
    question: &ContextQueryQuestion,
    keywords: &HashSet<String>,
    max_matches: usize,
    max_match_chars: usize,
) -> Vec<ContextQueryMatch> {
    let ranked_matches = build_ranked_query_matches(
        build_ranked_query_hits(candidates, keywords),
        max_matches,
        max_match_chars,
        "sqlite_ranked",
    );
    let fts_matches =
        build_sqlite_fts_matches(candidates, question, keywords, max_matches, max_match_chars)
            .unwrap_or_default();
    let semantic_matches = embedding_target
        .and_then(|target| {
            build_semantic_query_matches(
                paths,
                target,
                candidates,
                question,
                keywords,
                max_matches,
                max_match_chars,
            )
        })
        .unwrap_or_default();
    let mut combined = HashMap::<(String, String), ContextQueryMatch>::new();
    for (position, item) in ranked_matches.into_iter().enumerate() {
        let mut merged = item.clone();
        merged.score = merged
            .score
            .saturating_add(max_matches.saturating_sub(position).saturating_mul(7));
        merged.query_mode_used = "sqlite_hybrid".to_string();
        combined.insert(
            (merged.source_kind.clone(), merged.source_ref.clone()),
            merged,
        );
    }
    for (position, item) in fts_matches.into_iter().enumerate() {
        let key = (item.source_kind.clone(), item.source_ref.clone());
        let entry = combined.entry(key).or_insert_with(|| ContextQueryMatch {
            query_mode_used: "sqlite_hybrid".to_string(),
            ..item.clone()
        });
        entry.score = entry
            .score
            .saturating_add(item.score)
            .saturating_add(max_matches.saturating_sub(position).saturating_mul(9));
        if entry.excerpt.len() < item.excerpt.len() {
            entry.excerpt = item.excerpt.clone();
        }
        entry.query_mode_used = "sqlite_hybrid".to_string();
    }
    for (position, item) in semantic_matches.into_iter().enumerate() {
        let key = (item.source_kind.clone(), item.source_ref.clone());
        let entry = combined.entry(key).or_insert_with(|| ContextQueryMatch {
            query_mode_used: "sqlite_hybrid".to_string(),
            ..item.clone()
        });
        entry.score = entry
            .score
            .saturating_add(item.score)
            .saturating_add(max_matches.saturating_sub(position).saturating_mul(11));
        if entry.excerpt.len() < item.excerpt.len() {
            entry.excerpt = item.excerpt.clone();
        }
        entry.query_mode_used = "sqlite_hybrid".to_string();
    }
    let mut merged = combined.into_values().collect::<Vec<_>>();
    merged.sort_by(|left, right| right.score.cmp(&left.score));
    merged.into_iter().take(max_matches).collect()
}

fn build_context_query_answers(
    paths: &Paths,
    task: &TaskRecord,
    policy: &ContextOptimizationPolicy,
    query_contract: &ContextQueryToolContract,
    artifact: Option<&ContextPreparedArtifact>,
) -> Vec<ContextQueryAnswer> {
    let Some(artifact) = artifact else {
        return Vec::new();
    };
    if artifact.questions.is_empty() {
        return Vec::new();
    }
    let candidates = build_context_query_candidates(paths, task);
    let embedding_target = resolve_context_embedding_target(paths);
    artifact
        .questions
        .iter()
        .take(policy.max_questions)
        .map(|question| {
            let requested_query_mode = question.query_mode.trim().to_ascii_lowercase();
            let query_mode = if requested_query_mode.is_empty()
                || !query_contract
                    .allowed_query_modes
                    .iter()
                    .any(|mode| mode.eq_ignore_ascii_case(&requested_query_mode))
            {
                query_contract.default_query_mode.clone()
            } else {
                requested_query_mode
            };
            let mut keywords = extract_keywords(&format!(
                "{} {} {}",
                task.title, task.detail, question.question
            ));
            for keyword in &question.required_keywords {
                if !keyword.trim().is_empty() {
                    keywords.insert(keyword.trim().to_ascii_lowercase());
                }
            }
            let source_filters = question
                .source_kinds
                .iter()
                .map(|value| value.to_ascii_lowercase())
                .collect::<Vec<_>>();
            let max_matches = question
                .max_matches
                .unwrap_or(policy.max_matches_per_question)
                .min(policy.max_matches_per_question)
                .max(1);
            let filtered_candidates =
                filter_query_candidates(&candidates, &source_filters, query_contract);
            let matches = match query_mode.as_str() {
                "sqlite_fts" => build_sqlite_fts_matches(
                    &filtered_candidates,
                    question,
                    &keywords,
                    max_matches,
                    policy.max_match_chars,
                )
                .unwrap_or_else(|| {
                    build_ranked_query_matches(
                        build_ranked_query_hits(&filtered_candidates, &keywords),
                        max_matches,
                        policy.max_match_chars,
                        "sqlite_fts_fallback_ranked",
                    )
                }),
                "sqlite_hybrid" => build_hybrid_query_matches(
                    paths,
                    embedding_target.as_ref(),
                    &filtered_candidates,
                    question,
                    &keywords,
                    max_matches,
                    policy.max_match_chars,
                ),
                _ => build_ranked_query_matches(
                    build_ranked_query_hits(&filtered_candidates, &keywords),
                    max_matches,
                    policy.max_match_chars,
                    "sqlite_ranked",
                ),
            };
            ContextQueryAnswer {
                question: trim_chars(&question.question, 220),
                why: trim_chars(&question.why, 220),
                matches,
            }
        })
        .collect()
}

pub fn prepare_context_package(paths: &Paths, task: &TaskRecord) -> anyhow::Result<ContextPackage> {
    prepare_context_package_with_trigger(paths, task, ContextCompactionTrigger::Auto)
}

pub fn prepare_context_package_with_trigger(
    paths: &Paths,
    task: &TaskRecord,
    trigger: ContextCompactionTrigger,
) -> anyhow::Result<ContextPackage> {
    let policy = load_context_policy(paths);
    let optimization_policy = load_context_optimization_policy(paths);
    let query_contract = load_context_query_tool_contract(paths);
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
    let base_mode = choose_mode(&policy, &optimization_policy, task, None);
    let focus = load_focus_state(paths)?;
    let disk_headroom = inspect_runtime_disk_headroom(paths).ok();

    let keywords = extract_keywords(&format!("{} {}", task.title, task.detail));
    let checkpoint_limit = checkpoint_lookback_limit(&policy, &base_mode, &task.task_kind);
    let all_checkpoints = list_task_checkpoints(paths, task.id, checkpoint_limit)?;
    let prepared_context_artifact = latest_prepared_context_artifact(&all_checkpoints);
    let embedding_target = resolve_context_embedding_target(paths);
    let context_query_answers = build_context_query_answers(
        paths,
        task,
        &optimization_policy,
        &query_contract,
        prepared_context_artifact.as_ref(),
    );
    let active_phase = infer_context_optimization_phase(
        task,
        prepared_context_artifact.as_ref(),
        &context_query_answers,
    );
    let mode = choose_mode(
        &policy,
        &optimization_policy,
        task,
        Some(active_phase.as_str()),
    );
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
    let checkpoints = all_checkpoints
        .iter()
        .take(mode.recent_task_checkpoints)
        .cloned()
        .collect::<Vec<_>>();
    let preparation_state = build_context_preparation_state(
        &optimization_policy,
        &active_phase,
        &all_checkpoints,
        prepared_context_artifact.as_ref(),
    );
    let turn_signals = select_turn_signals(list_turn_signals_for_task(paths, task.id, 6)?, 3);
    let exec_sessions = select_exec_sessions(snapshot_sessions().unwrap_or_default(), task.id, 6);
    let mut raw_inclusions = build_raw_inclusions(task, &checkpoints, &bios_dialogue, &mode);
    append_parent_task_inclusions(paths, task, &mut raw_inclusions);
    if installation_bootstrap.status != "unconfigured" {
        let bootstrap_json = serde_json::to_string_pretty(&installation_bootstrap)
            .unwrap_or_else(|_| "{}".to_string());
        raw_inclusions.push(ContextRawInclusion {
            source_kind: "installation_bootstrap".to_string(),
            source_ref: paths.installation_bootstrap_path.display().to_string(),
            content: trim_chars(&bootstrap_json, 1600),
        });
    }
    append_definition_of_done_inclusions(paths, &mut raw_inclusions);
    append_installation_tool_smoke_inclusions(paths, task, &mut raw_inclusions);
    if task_should_get_codex_command_exec_guidance(task) {
        append_codex_command_exec_inclusions(paths, &mut raw_inclusions);
    }
    append_owner_operation_inclusions(paths, task, &mut raw_inclusions);
    append_recent_owner_outcome_inclusions(paths, task, &mut raw_inclusions);
    append_current_task_machine_evidence_inclusion(
        Some(paths),
        task.id,
        &all_checkpoints,
        &mut raw_inclusions,
    );
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
            operating_rule: "Kleinhirn is the free local default. For screenshot-heavy or UI-perception-heavy browser work, first consider a vision-capable local Qwen3.5 kleinhirn. External grosshirn is only a temporary per-task boost: you decide when the cost is justified, keep local kleinhirn fallback ready, and release the boost again after difficult phases.".to_string(),
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
        context_optimization: build_context_optimization_summary(
            &optimization_policy,
            active_phase,
        ),
        context_query_contract: build_context_query_contract_summary(
            &query_contract,
            embedding_target.as_ref(),
        ),
        preparation_contract: if task.task_kind == "context_preparation" {
            Some(build_context_preparation_contract(
                &optimization_policy,
                &infer_context_optimization_phase(
                    task,
                    prepared_context_artifact.as_ref(),
                    &context_query_answers,
                ),
            ))
        } else {
            None
        },
        preparation_review_contract: if task.task_kind == "context_preparation" {
            Some(build_context_preparation_review_contract(
                &optimization_policy,
                &infer_context_optimization_phase(
                    task,
                    prepared_context_artifact.as_ref(),
                    &context_query_answers,
                ),
            ))
        } else {
            None
        },
        preparation_state,
        context_query_answers,
        prepared_context_artifact,
        context_distillation: None,
        compact_controller: None,
        retrieval_notes: vec![
            "Use this package as the active starting context for exactly one bounded task run.".to_string(),
            "Do not pull the entire history into context; raw past should be reloaded only deliberately.".to_string(),
            "If an older decision is missing, request targeted historical reload instead of speculating broadly.".to_string(),
            "Normal context maintenance, compaction, and reload are your agentic decisions; the kernel intervenes only at a hard overflow boundary.".to_string(),
            "The context optimization loop is separate from task execution: first ask high-value questions, then retrieve narrowly, then rewrite the final context package block by block.".to_string(),
            "`contextQueryContract` is the formal schema for SQLite-backed context search. Follow it instead of inventing ad-hoc broad retrieval.".to_string(),
            "For context preparation, `contextQueryAnswers` are evidence candidates from SQLite-backed runtime state, not prompt text. Rewrite the final blocks yourself token by token.".to_string(),
            if embedding_target.is_some() {
                "`sqlite_hybrid` may use semantic retrieval through the configured mistral.rs embedding endpoint in addition to lexical ranking and SQLite FTS.".to_string()
            } else {
                "`sqlite_hybrid` currently falls back to lexical ranking plus SQLite FTS because no embedding runtime is configured.".to_string()
            },
            "When `contextDistillation` exists, use its `activeFocus` first, then its continuity narrative and anchors; treat broader package state as support and verification.".to_string(),
            "When `compactController` exists, treat it as the wrapper-facing bridge summary for continuity, reprioritization, and model routing at this compaction boundary.".to_string(),
            "`contextDistillation.continuityArtifacts` carries the concrete continuity payload for the next turn: must-touch files, contracts, sessions, checkpoints, and similar artifacts.".to_string(),
            "Only context written into `preparedContextArtifact.blocks` or distilled from it should become the direct execution handoff; do not dump irrelevant retrieval matches into the active run context.".to_string(),
            "Every prepared block should carry `evidenceRefs`; blocks without provenance are not ready for execution.".to_string(),
            "Treat `contextOptimization.surfaces` as CTO memory surfaces to activate or leave quiet deliberately; they are not prompt-writing topics.".to_string(),
            "Use the signal asymmetry on purpose: many fine-grained negative signals should diagnose missing, stale, conflicting, or noisy context, while fewer broader positive signals only confirm that a surface is truly in good shape.".to_string(),
            "`contextDistillation.historicalRetrievalRefs` lists the exact SQLite/embedding refs that can be reloaded later without dragging the whole past back in.".to_string(),
            "Exec sessions are real running terminal contexts. If a matching session already exists, you can continue it deliberately by session ID instead of inventing new shell state.".to_string(),
            "If GPU, VRAM, or `mistralrs tune` evidence is missing for a local kleinhirn upgrade, you may set `systemCensusAction=run`.".to_string(),
            "If installation bootstrap appears in context, treat it as real owner and installer guidance for early communication capability and channel planning.".to_string(),
            "Repo skills under `.agents/skills` are your persistent self-extension surface. If you write a new skill there, it will reappear in the skill catalog on the next turn.".to_string(),
            "If you build a new tool, also write an operations skill with the concrete commands, paths, and error boundaries for later turns.".to_string(),
            "`hostSurvival` gives you current disk headroom as a CTO signal. Use it agentically: if storage gets tight, prioritize bounded inspection, capacity planning, and safe cleanup work yourself instead of expanding blindly.".to_string(),
            "`learningSummaries` is your condensed high-level memory. Use `relevantLearningEntries` when you need to apply the details of an earlier learning concretely again.".to_string(),
            "`peopleWorkingSet` and `relevantPeople` hold highly condensed people paths, conversation notes, and mail references so you do not treat humans like stateless tickets.".to_string(),
            "`pendingProactiveContacts` are only drafts or review cases. Never claim that a proactive suggestion has already been sent until a separate dispatch has really happened.".to_string(),
            "If you gain a new durable learning in this bounded step, return it as `learningEntries` instead of hiding it only in free text.".to_string(),
        ],
    };

    let mut package = package;
    package.context_distillation = build_context_distillation(&package);
    compact_context_package_for_preparation(&mut package);
    compact_context_package_for_execution_handoff(&mut package);
    package.compact_controller = build_compact_controller(paths, task, &package, trigger);
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
    entries.retain(|entry| entry.session_id.contains(&task_marker));
    entries.sort_by(|left, right| {
        let left_active = left.status == "active";
        let right_active = right.status == "active";
        right_active
            .cmp(&left_active)
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
    optimization_policy: &ContextOptimizationPolicy,
    task: &TaskRecord,
    preparation_phase: Option<&str>,
) -> ContextModePolicy {
    let task_kind = task.task_kind.to_lowercase();
    let source_channel = task.source_channel.to_lowercase();
    let lowered_detail = task.detail.to_lowercase();
    let mode_name = if task_kind == "context_preparation" {
        if let Some(phase) = preparation_phase
            && let Some(override_name) = context_phase_policy(optimization_policy, phase)
                .and_then(|phase_policy| phase_policy.context_mode_override.as_deref())
            && policy.modes.iter().any(|mode| mode.mode == override_name)
        {
            override_name
        } else {
            "preparation"
        }
    } else if policy
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
                    budget_hint: 32768,
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
    select_by_keywords(
        entries,
        |entry| format!("{} {}", entry.speaker, entry.message),
        keywords,
        limit,
    )
}

fn select_memory_items(
    entries: Vec<MemoryItemRecord>,
    keywords: &HashSet<String>,
    limit: usize,
) -> Vec<MemoryItemRecord> {
    select_by_keywords(
        entries,
        |entry| format!("{} {}", entry.summary, entry.detail),
        keywords,
        limit,
    )
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
            score += usize::try_from(entry.interaction_count.max(0))
                .unwrap_or_default()
                .min(12);
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
    ranked
        .into_iter()
        .take(limit)
        .map(|(_, _, entry)| entry)
        .collect()
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

fn append_parent_task_inclusions(
    paths: &Paths,
    task: &TaskRecord,
    raw_inclusions: &mut Vec<ContextRawInclusion>,
) {
    let Some(parent_task_id) = task.parent_task_id else {
        return;
    };
    let Ok(Some(parent_task)) = load_task_by_id(paths, parent_task_id) else {
        return;
    };
    let parent_snapshot = format!(
        "Parent Task #{id}\nTitle: {title}\nKind: {kind}\nStatus: {status}\nPriority: {priority}\nDetail:\n{detail}",
        id = parent_task.id,
        title = parent_task.title,
        kind = parent_task.task_kind,
        status = parent_task.status,
        priority = parent_task.priority_score,
        detail = trim_chars(&parent_task.detail, 1200),
    );
    raw_inclusions.push(ContextRawInclusion {
        source_kind: "parent_task".to_string(),
        source_ref: format!("task:{}", parent_task.id),
        content: parent_snapshot,
    });
    for checkpoint in list_task_checkpoints(paths, parent_task.id, 2)
        .unwrap_or_default()
        .into_iter()
        .take(2)
    {
        raw_inclusions.push(ContextRawInclusion {
            source_kind: "parent_task_checkpoint".to_string(),
            source_ref: format!("task:{}:{}", parent_task.id, checkpoint.checkpoint_kind),
            content: trim_chars(&checkpoint.detail, 700),
        });
    }
}

fn append_codex_command_exec_inclusions(
    paths: &Paths,
    raw_inclusions: &mut Vec<ContextRawInclusion>,
) {
    push_matching_raw_inclusions(
        raw_inclusions,
        "repo_operation_skill",
        paths.root.join(".agents/skills"),
        |path| {
            path.file_name().and_then(|name| name.to_str()) == Some("SKILL.md")
                && path
                    .parent()
                    .and_then(|parent| parent.file_name())
                    .and_then(|name| name.to_str())
                    == Some("codex-command-exec-operations")
        },
        1,
        2200,
    );
    push_matching_raw_inclusions(
        raw_inclusions,
        "system_capability_contract",
        paths.system_dir.clone(),
        |path| {
            path.file_name().and_then(|name| name.to_str())
                == Some("codex-command-exec-capability-policy.json")
        },
        1,
        2600,
    );
}

fn task_should_get_codex_command_exec_guidance(task: &TaskRecord) -> bool {
    matches!(
        task.task_kind.as_str(),
        "owner_interrupt" | "workspace_repair" | "tool_exploration"
    )
}

fn owner_interrupt_needs_workspace_execution_guidance(task: &TaskRecord) -> bool {
    if task.task_kind != "owner_interrupt" {
        return false;
    }
    let haystack = format!("{} {}", task.title, task.detail).to_ascii_lowercase();
    [
        "c++",
        "cpp",
        ".cpp",
        ".hpp",
        ".h",
        "cmake",
        "console app",
        "konsolenanwendung",
        "application",
        "anwendung",
        "implement",
        "implementation",
        "entwick",
        "code",
        "repo",
        "repository",
        "workspace",
        "build",
        "compile",
        "test",
        "thread-safe",
        "threadsicher",
        "race condition",
        "persistent",
        "persisten",
    ]
    .iter()
    .any(|needle| haystack.contains(needle))
}

fn append_owner_operation_inclusions(
    paths: &Paths,
    task: &TaskRecord,
    raw_inclusions: &mut Vec<ContextRawInclusion>,
) {
    if task.task_kind != "owner_interrupt" {
        return;
    }
    let keywords = extract_keywords(&format!("{} {}", task.title, task.detail));
    push_relevant_file_raw_inclusions(
        raw_inclusions,
        "repo_operation_skill",
        paths.root.join(".agents/skills"),
        |path| {
            path.file_name().and_then(|name| name.to_str()) == Some("SKILL.md")
                && path
                    .parent()
                    .and_then(|parent| parent.file_name())
                    .and_then(|name| name.to_str())
                    .map(|name| name.ends_with("-operations"))
                    .unwrap_or(false)
        },
        &keywords,
        4,
        1800,
    );
    push_relevant_file_raw_inclusions(
        raw_inclusions,
        "system_capability_contract",
        paths.system_dir.clone(),
        |path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.ends_with("-capability-policy.json"))
                .unwrap_or(false)
        },
        &keywords,
        2,
        2200,
    );
    if owner_interrupt_needs_workspace_execution_guidance(task) {
        push_matching_raw_inclusions(
            raw_inclusions,
            "repo_operation_skill",
            paths.root.join(".agents/skills"),
            |path| {
                path.file_name().and_then(|name| name.to_str()) == Some("SKILL.md")
                    && path
                        .parent()
                        .and_then(|parent| parent.file_name())
                        .and_then(|name| name.to_str())
                        == Some("workspace-execution-operations")
            },
            1,
            2200,
        );
        push_matching_raw_inclusions(
            raw_inclusions,
            "system_capability_contract",
            paths.system_dir.clone(),
            |path| {
                path.file_name().and_then(|name| name.to_str())
                    == Some("workspace-execution-capability-policy.json")
            },
            1,
            2600,
        );
    }
}

fn append_definition_of_done_inclusions(
    paths: &Paths,
    raw_inclusions: &mut Vec<ContextRawInclusion>,
) {
    push_matching_raw_inclusions(
        raw_inclusions,
        "task_definition_of_done_policy",
        paths.system_dir.clone(),
        |path| {
            path.file_name().and_then(|name| name.to_str())
                == Some("task-definition-of-done-policy.json")
        },
        1,
        2200,
    );
}

fn append_installation_tool_smoke_inclusions(
    paths: &Paths,
    task: &TaskRecord,
    raw_inclusions: &mut Vec<ContextRawInclusion>,
) {
    if !matches!(
        task.task_kind.as_str(),
        "bootstrap_runtime_guard"
            | "tool_exploration"
            | "installation_bootstrap"
            | "channel_expansion"
    ) {
        return;
    }
    push_matching_raw_inclusions(
        raw_inclusions,
        "installation_tool_smoke_resource",
        paths.bootstrap_dir.clone(),
        |path| {
            path.file_name().and_then(|name| name.to_str())
                == Some("installation-tool-smoke-resource.json")
        },
        1,
        3200,
    );
}

fn append_recent_owner_outcome_inclusions(
    paths: &Paths,
    task: &TaskRecord,
    raw_inclusions: &mut Vec<ContextRawInclusion>,
) {
    if task.task_kind != "owner_interrupt" {
        return;
    }
    let recent_outcomes = list_recent_task_outcomes(paths, task.id, 4).unwrap_or_default();
    for outcome in recent_outcomes.into_iter().take(3) {
        let mut content = String::new();
        if let Some(summary) = outcome.last_checkpoint_summary.as_deref() {
            if !summary.trim().is_empty() {
                content.push_str("Summary: ");
                content.push_str(summary.trim());
            }
        }
        if let Some(output) = outcome.last_output.as_deref() {
            if !output.trim().is_empty() {
                if !content.is_empty() {
                    content.push_str("\n\n");
                }
                content.push_str("Output:\n");
                content.push_str(output.trim());
            }
        }
        if content.trim().is_empty() {
            continue;
        }
        raw_inclusions.push(ContextRawInclusion {
            source_kind: "recent_task_outcome".to_string(),
            source_ref: format!("task:{}", outcome.id),
            content: trim_chars(&content, 1200),
        });
    }
}

fn extract_checkpoint_detail_value(detail: &str, prefix: &str) -> Option<String> {
    detail
        .lines()
        .find_map(|line| line.trim().strip_prefix(prefix))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn is_notable_workspace_anchor_line(line: &str) -> bool {
    let trimmed = line.trim();
    if trimmed.is_empty() || trimmed.starts_with("---") {
        return false;
    }
    let lower = trimmed.to_ascii_lowercase();
    trimmed.starts_with("PWD=")
        || trimmed.contains('/')
        || trimmed.contains('\\')
        || lower.contains(".cpp")
        || lower.contains(".cc")
        || lower.contains(".cxx")
        || lower.contains(".h")
        || lower.contains(".hpp")
        || lower.contains(".rs")
        || lower.contains("cmakelists.txt")
        || lower.contains("makefile")
        || lower.contains("cargo.toml")
        || lower.contains("package.json")
}

fn extract_notable_workspace_anchor_lines(detail: &str, limit: usize) -> Vec<String> {
    let mut results = Vec::new();
    let mut seen = HashSet::new();
    let mut in_stdout = false;
    for line in detail.lines() {
        let trimmed = line.trim();
        if trimmed == "STDOUT:" || trimmed.eq_ignore_ascii_case("--- stdout ---") {
            in_stdout = true;
            continue;
        }
        if !in_stdout {
            continue;
        }
        if trimmed == "STDERR:" || trimmed.eq_ignore_ascii_case("--- stderr ---") {
            break;
        }
        if !is_notable_workspace_anchor_line(trimmed) {
            continue;
        }
        let normalized = trim_chars(trimmed, 120);
        if seen.insert(normalized.clone()) {
            results.push(normalized);
        }
        if results.len() >= limit {
            break;
        }
    }
    if !results.is_empty() {
        return results;
    }
    for line in detail.lines() {
        let trimmed = line.trim();
        if !is_notable_workspace_anchor_line(trimmed) {
            continue;
        }
        let normalized = trim_chars(trimmed, 120);
        if seen.insert(normalized.clone()) {
            results.push(normalized);
        }
        if results.len() >= limit {
            break;
        }
    }
    results
}

fn summarize_current_task_machine_evidence(
    task_id: i64,
    checkpoints: &[TaskCheckpointRecord],
) -> Option<ContextRawInclusion> {
    let checkpoint = checkpoints
        .iter()
        .find(|entry| checkpoint_contains_machine_evidence(&entry.summary, &entry.detail))?;
    let mut parts = vec![format!(
        "Latest verified machine step for task #{}: {}",
        task_id,
        trim_chars(&checkpoint.summary, 180)
    )];
    let workdir = extract_checkpoint_detail_value(&checkpoint.detail, "Workdir: ")
        .or_else(|| extract_checkpoint_detail_value(&checkpoint.detail, "CWD: "));
    if let Some(workdir) = workdir {
        parts.push(format!("Workdir: {}", trim_chars(&workdir, 160)));
    }
    let command = extract_checkpoint_detail_value(&checkpoint.detail, "Command: ")
        .or_else(|| extract_checkpoint_detail_value(&checkpoint.detail, "Requested command: "));
    if let Some(command) = command {
        parts.push(format!("Command: {}", trim_chars(&command, 220)));
    }
    let observed = extract_notable_workspace_anchor_lines(&checkpoint.detail, 4);
    if !observed.is_empty() {
        parts.push(format!("Observed anchors: {}", observed.join(" | ")));
    }
    Some(ContextRawInclusion {
        source_kind: "current_task_machine_evidence".to_string(),
        source_ref: format!("task:{}:machine_anchor", task_id),
        content: trim_chars(&parts.join("\n"), 1400),
    })
}

fn load_latest_machine_evidence_checkpoint(
    paths: &Paths,
    task_id: i64,
) -> Option<TaskCheckpointRecord> {
    let conn = Connection::open(&paths.runtime_db_path).ok()?;
    let mut stmt = conn
        .prepare(
            "SELECT task_id, created_at, checkpoint_kind, summary, detail
             FROM task_checkpoints
             WHERE task_id = ?1
             ORDER BY id DESC",
        )
        .ok()?;
    let mut rows = stmt.query(params![task_id]).ok()?;
    loop {
        let Some(row) = rows.next().ok()? else {
            return None;
        };
        let checkpoint = TaskCheckpointRecord {
            task_id: row.get(0).ok()?,
            created_at: row.get(1).ok()?,
            checkpoint_kind: row.get(2).ok()?,
            summary: row.get(3).ok()?,
            detail: row.get(4).ok()?,
        };
        if checkpoint_contains_machine_evidence(&checkpoint.summary, &checkpoint.detail) {
            return Some(checkpoint);
        }
    }
}

fn append_current_task_machine_evidence_inclusion(
    paths: Option<&Paths>,
    task_id: i64,
    checkpoints: &[TaskCheckpointRecord],
    raw_inclusions: &mut Vec<ContextRawInclusion>,
) {
    if let Some(anchor) = summarize_current_task_machine_evidence(task_id, checkpoints) {
        raw_inclusions.push(anchor);
        return;
    }
    let Some(paths) = paths else {
        return;
    };
    let Some(checkpoint) = load_latest_machine_evidence_checkpoint(paths, task_id) else {
        return;
    };
    if let Some(anchor) = summarize_current_task_machine_evidence(task_id, &[checkpoint]) {
        raw_inclusions.push(anchor);
    }
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

fn push_matching_raw_inclusions(
    raw_inclusions: &mut Vec<ContextRawInclusion>,
    source_kind: &str,
    root: std::path::PathBuf,
    matcher: impl Fn(&std::path::Path) -> bool,
    limit_files: usize,
    limit_chars: usize,
) {
    let Ok(entries) = fs::read_dir(&root) else {
        return;
    };
    let mut matches = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let Ok(children) = fs::read_dir(&path) else {
                continue;
            };
            for child in children.flatten() {
                let child_path = child.path();
                if matcher(&child_path) {
                    matches.push(child_path);
                }
            }
        } else if matcher(&path) {
            matches.push(path);
        }
    }
    matches.sort();
    for path in matches.into_iter().take(limit_files) {
        push_file_raw_inclusion(raw_inclusions, source_kind, path, limit_chars);
    }
}

fn push_relevant_file_raw_inclusions(
    raw_inclusions: &mut Vec<ContextRawInclusion>,
    source_kind: &str,
    root: std::path::PathBuf,
    matcher: impl Fn(&std::path::Path) -> bool,
    keywords: &HashSet<String>,
    limit_files: usize,
    limit_chars: usize,
) {
    let Ok(entries) = fs::read_dir(&root) else {
        return;
    };
    let mut matches = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            let Ok(children) = fs::read_dir(&path) else {
                continue;
            };
            for child in children.flatten() {
                let child_path = child.path();
                if matcher(&child_path) {
                    let content = fs::read_to_string(&child_path).unwrap_or_default();
                    let haystack = format!("{} {}", child_path.display(), content);
                    matches.push((relevance_score(&haystack, keywords), child_path));
                }
            }
        } else if matcher(&path) {
            let content = fs::read_to_string(&path).unwrap_or_default();
            let haystack = format!("{} {}", path.display(), content);
            matches.push((relevance_score(&haystack, keywords), path));
        }
    }
    matches.sort_by(|left, right| right.0.cmp(&left.0).then(left.1.cmp(&right.1)));
    for (_, path) in matches.into_iter().take(limit_files) {
        push_file_raw_inclusion(raw_inclusions, source_kind, path, limit_chars);
    }
}

fn build_rationale(task: &TaskRecord, mode: &str, keyword_count: usize) -> String {
    format!(
        "context_mode={} for task {} ({}) via {} with {} extracted topic keywords",
        mode, task.id, task.task_kind, task.source_channel, keyword_count
    )
}

fn allowed_next_modes(policy: &ModeSystemPolicy, current_mode: &str) -> Vec<String> {
    let mut modes: Vec<String> = policy
        .transitions
        .iter()
        .filter(|rule| rule.from_mode == current_mode)
        .map(|rule| rule.to_mode.clone())
        .collect();
    if matches!(
        current_mode,
        "execute_task" | "self_preservation" | "recovery" | "historical_research"
    ) && !modes.iter().any(|mode| mode == current_mode)
    {
        modes.insert(0, current_mode.to_string());
    }
    if modes.is_empty() {
        vec!["reprioritize".to_string(), "idle".to_string()]
    } else {
        modes
    }
}

fn checkpoint_lookback_limit(
    policy: &crate::contracts::ContextPolicy,
    base_mode: &ContextModePolicy,
    task_kind: &str,
) -> usize {
    if task_kind == "context_preparation" {
        policy
            .modes
            .iter()
            .map(|mode| mode.recent_task_checkpoints)
            .max()
            .unwrap_or(base_mode.recent_task_checkpoints)
    } else {
        base_mode.recent_task_checkpoints.max(4)
    }
}

fn extract_keywords(text: &str) -> HashSet<String> {
    const STOPWORDS: &[&str] = &[
        "this", "that", "with", "from", "have", "your", "task", "tasks", "oder", "aber", "dass",
        "eine", "einen", "einer", "eines", "fuer", "ueber", "about", "into", "then", "when",
        "will", "must", "should", "und", "der", "die", "das",
    ];
    let tokens: Vec<String> = text
        .split(|c: char| !c.is_alphanumeric())
        .map(|part| part.trim().to_lowercase())
        .filter(|part| part.len() >= 3)
        .filter(|part| !STOPWORDS.contains(&part.as_str()))
        .collect();
    let mut keywords = HashSet::new();
    for token in tokens.iter().take(32) {
        keywords.insert(token.clone());
    }
    for window in tokens.windows(2).take(16) {
        keywords.insert(format!("{} {}", window[0], window[1]));
    }
    keywords
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
        std::fs::create_dir_all(root.join(".agents/skills/workspace-execution-operations"))
            .unwrap();
        std::fs::create_dir_all(root.join(".agents/skills/codex-command-exec-operations")).unwrap();
        std::fs::create_dir_all(root.join("contracts/system")).unwrap();
        root
    }

    fn sample_paths(root: &PathBuf) -> Paths {
        Paths {
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
            installation_bootstrap_path: root
                .join("contracts/bootstrap/installation-bootstrap.json"),
            context_policy_path: root.join("contracts/context/context-policy.json"),
            context_optimization_policy_path: root
                .join("contracts/context/context-optimization-policy.json"),
            context_query_tool_contract_path: root
                .join("contracts/context/context-query-tool-contract.json"),
            context_governance_policy_path: root
                .join("contracts/context/context-governance-policy.json"),
            mode_system_policy_path: root.join("contracts/system/mode-system-policy.json"),
            loop_safety_policy_path: root.join("contracts/system/loop-safety-policy.json"),
            execution_authority_policy_path: root
                .join("contracts/system/execution-authority-policy.json"),
            browser_engine_policy_path: root.join("contracts/browser/browser-engine-policy.json"),
            browser_capability_policy_path: root
                .join("contracts/browser/browser-capability-policy.json"),
            browser_subworker_policy_path: root
                .join("contracts/browser/browser-subworker-policy.json"),
            self_preservation_state_path: root
                .join("contracts/system/self-preservation-state.json"),
            origin_story_path: root.join("contracts/history/origin-story.md"),
            creation_ledger_path: root.join("contracts/history/creation-ledger.md"),
            boot_log_path: root.join("runtime/boot_log.jsonl"),
            agent_state_path: root.join("runtime/state/agent_state.json"),
            system_census_path: root.join("runtime/state/system_census.json"),
            browser_engine_state_path: root.join("runtime/state/browser_engine_state.json"),
            runtime_db_path: root.join("runtime/cto_agent.db"),
            attach_socket_path: root.join("runtime/cto-agent.sock"),
            runtime_lock_path: root.join("runtime/cto-agent.lock"),
            pending_hard_reset_report_path: root
                .join("runtime/recovery/pending-hard-reset-report.json"),
            certs_dir: root.join("runtime/certs"),
            tls_cert_path: root.join("runtime/certs/localhost.crt"),
            tls_key_path: root.join("runtime/certs/localhost.key"),
        }
    }

    #[test]
    fn owner_interrupt_gets_owner_operation_context_without_prompt_matching() {
        let task = sample_task(
            "Please execute the requested host step",
            "Trusted owner request with no keyboard words in the prompt body.",
        );
        let mut raw_inclusions = Vec::new();
        let root = temp_root("generic_owner_ops");
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
            installation_bootstrap_path: root
                .join("contracts/bootstrap/installation-bootstrap.json"),
            context_policy_path: root.join("contracts/context/context-policy.json"),
            context_optimization_policy_path: root
                .join("contracts/context/context-optimization-policy.json"),
            context_query_tool_contract_path: root
                .join("contracts/context/context-query-tool-contract.json"),
            context_governance_policy_path: root
                .join("contracts/context/context-governance-policy.json"),
            mode_system_policy_path: root.join("contracts/system/mode-system-policy.json"),
            loop_safety_policy_path: root.join("contracts/system/loop-safety-policy.json"),
            execution_authority_policy_path: root
                .join("contracts/system/execution-authority-policy.json"),
            browser_engine_policy_path: root.join("contracts/browser/browser-engine-policy.json"),
            browser_capability_policy_path: root
                .join("contracts/browser/browser-capability-policy.json"),
            browser_subworker_policy_path: root
                .join("contracts/browser/browser-subworker-policy.json"),
            self_preservation_state_path: root
                .join("contracts/system/self-preservation-state.json"),
            origin_story_path: root.join("contracts/history/origin-story.md"),
            creation_ledger_path: root.join("contracts/history/creation-ledger.md"),
            boot_log_path: root.join("runtime/boot_log.jsonl"),
            agent_state_path: root.join("runtime/state/agent_state.json"),
            system_census_path: root.join("runtime/state/system_census.json"),
            browser_engine_state_path: root.join("runtime/state/browser_engine_state.json"),
            runtime_db_path: root.join("runtime/cto_agent.db"),
            attach_socket_path: root.join("runtime/cto-agent.sock"),
            runtime_lock_path: root.join("runtime/cto-agent.lock"),
            pending_hard_reset_report_path: root
                .join("runtime/recovery/pending-hard-reset-report.json"),
            certs_dir: root.join("runtime/certs"),
            tls_cert_path: root.join("runtime/certs/localhost.crt"),
            tls_key_path: root.join("runtime/certs/localhost.key"),
        };
        append_owner_operation_inclusions(&paths, &task, &mut raw_inclusions);
        assert_eq!(raw_inclusions.len(), 2);
        assert_eq!(raw_inclusions[0].source_kind, "repo_operation_skill");
        assert_eq!(raw_inclusions[1].source_kind, "system_capability_contract");
    }

    #[test]
    fn owner_interrupt_includes_operations_skill_and_contracts() {
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
            installation_bootstrap_path: root
                .join("contracts/bootstrap/installation-bootstrap.json"),
            context_policy_path: root.join("contracts/context/context-policy.json"),
            context_optimization_policy_path: root
                .join("contracts/context/context-optimization-policy.json"),
            context_query_tool_contract_path: root
                .join("contracts/context/context-query-tool-contract.json"),
            context_governance_policy_path: root
                .join("contracts/context/context-governance-policy.json"),
            mode_system_policy_path: root.join("contracts/system/mode-system-policy.json"),
            loop_safety_policy_path: root.join("contracts/system/loop-safety-policy.json"),
            execution_authority_policy_path: root
                .join("contracts/system/execution-authority-policy.json"),
            browser_engine_policy_path: root.join("contracts/browser/browser-engine-policy.json"),
            browser_capability_policy_path: root
                .join("contracts/browser/browser-capability-policy.json"),
            browser_subworker_policy_path: root
                .join("contracts/browser/browser-subworker-policy.json"),
            self_preservation_state_path: root
                .join("contracts/system/self-preservation-state.json"),
            origin_story_path: root.join("contracts/history/origin-story.md"),
            creation_ledger_path: root.join("contracts/history/creation-ledger.md"),
            boot_log_path: root.join("runtime/boot_log.jsonl"),
            agent_state_path: root.join("runtime/state/agent_state.json"),
            system_census_path: root.join("runtime/state/system_census.json"),
            browser_engine_state_path: root.join("runtime/state/browser_engine_state.json"),
            runtime_db_path: root.join("runtime/cto_agent.db"),
            attach_socket_path: root.join("runtime/cto-agent.sock"),
            runtime_lock_path: root.join("runtime/cto-agent.lock"),
            pending_hard_reset_report_path: root
                .join("runtime/recovery/pending-hard-reset-report.json"),
            certs_dir: root.join("runtime/certs"),
            tls_cert_path: root.join("runtime/certs/localhost.crt"),
            tls_key_path: root.join("runtime/certs/localhost.key"),
        };
        let task = sample_task(
            "Any trusted owner action",
            "No specific host wording required.",
        );
        let mut raw_inclusions = Vec::new();
        append_owner_operation_inclusions(&paths, &task, &mut raw_inclusions);
        assert_eq!(raw_inclusions.len(), 2);
        assert_eq!(raw_inclusions[0].source_kind, "repo_operation_skill");
        assert_eq!(raw_inclusions[1].source_kind, "system_capability_contract");
        assert!(raw_inclusions[1].content.contains("keyboard"));
    }

    #[test]
    fn owner_interrupt_prefers_relevant_operations_skill() {
        let root = temp_root("raw_inclusion_relevance");
        std::fs::write(
            root.join(".agents/skills/host-keyboard-operations/SKILL.md"),
            "# Host Keyboard Operations\nUse this skill for keyboard layout changes.\n",
        )
        .unwrap();
        std::fs::write(
            root.join(".agents/skills/workspace-execution-operations/SKILL.md"),
            "# Workspace Execution Operations\nReuse execSessionAction for multi-step workspace and repo work.\n",
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
            installation_bootstrap_path: root
                .join("contracts/bootstrap/installation-bootstrap.json"),
            context_policy_path: root.join("contracts/context/context-policy.json"),
            context_optimization_policy_path: root
                .join("contracts/context/context-optimization-policy.json"),
            context_query_tool_contract_path: root
                .join("contracts/context/context-query-tool-contract.json"),
            context_governance_policy_path: root
                .join("contracts/context/context-governance-policy.json"),
            mode_system_policy_path: root.join("contracts/system/mode-system-policy.json"),
            loop_safety_policy_path: root.join("contracts/system/loop-safety-policy.json"),
            execution_authority_policy_path: root
                .join("contracts/system/execution-authority-policy.json"),
            browser_engine_policy_path: root.join("contracts/browser/browser-engine-policy.json"),
            browser_capability_policy_path: root
                .join("contracts/browser/browser-capability-policy.json"),
            browser_subworker_policy_path: root
                .join("contracts/browser/browser-subworker-policy.json"),
            self_preservation_state_path: root
                .join("contracts/system/self-preservation-state.json"),
            origin_story_path: root.join("contracts/history/origin-story.md"),
            creation_ledger_path: root.join("contracts/history/creation-ledger.md"),
            boot_log_path: root.join("runtime/boot_log.jsonl"),
            agent_state_path: root.join("runtime/state/agent_state.json"),
            system_census_path: root.join("runtime/state/system_census.json"),
            browser_engine_state_path: root.join("runtime/state/browser_engine_state.json"),
            runtime_db_path: root.join("runtime/cto_agent.db"),
            attach_socket_path: root.join("runtime/cto-agent.sock"),
            runtime_lock_path: root.join("runtime/cto-agent.lock"),
            pending_hard_reset_report_path: root
                .join("runtime/recovery/pending-hard-reset-report.json"),
            certs_dir: root.join("runtime/certs"),
            tls_cert_path: root.join("runtime/certs/localhost.crt"),
            tls_key_path: root.join("runtime/certs/localhost.key"),
        };
        let task = sample_task(
            "Continue workspace implementation",
            "Reuse the repo workspace evidence and continue implementation without replaying the same edit command.",
        );
        let mut raw_inclusions = Vec::new();
        append_owner_operation_inclusions(&paths, &task, &mut raw_inclusions);
        assert!(!raw_inclusions.is_empty());
        assert_eq!(raw_inclusions[0].source_kind, "repo_operation_skill");
        assert!(
            raw_inclusions[0]
                .content
                .contains("Workspace Execution Operations")
        );
    }

    #[test]
    fn owner_interrupt_prefers_relevant_workspace_contract() {
        let root = temp_root("workspace_contract_relevance");
        std::fs::write(
            root.join("contracts/system/host-keyboard-capability-policy.json"),
            "{\"version\":1,\"purpose\":\"keyboard layout change\"}",
        )
        .unwrap();
        std::fs::write(
            root.join("contracts/system/workspace-execution-capability-policy.json"),
            "{\"version\":1,\"purpose\":\"repo workspace implementation and build verification\"}",
        )
        .unwrap();
        let paths = sample_paths(&root);
        let task = sample_task(
            "Continue workspace implementation",
            "Reuse repo evidence and continue the C++ workspace implementation with bounded verification.",
        );
        let mut raw_inclusions = Vec::new();
        append_owner_operation_inclusions(&paths, &task, &mut raw_inclusions);
        let contract = raw_inclusions
            .iter()
            .find(|item| item.source_kind == "system_capability_contract")
            .expect("expected a relevant system capability contract");
        assert!(contract.content.contains("repo workspace implementation"));
    }

    #[test]
    fn coding_owner_interrupt_forces_workspace_guidance_without_workspace_wording() {
        let root = temp_root("coding_owner_interrupt_workspace_guidance");
        std::fs::write(
            root.join(".agents/skills/workspace-execution-operations/SKILL.md"),
            "# Workspace Execution Operations\nReuse execSessionAction for multi-step workspace and repo work.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("contracts/system/workspace-execution-capability-policy.json"),
            "{\"version\":1,\"purpose\":\"repo-local workspace implementation and build verification\"}",
        )
        .unwrap();
        let paths = sample_paths(&root);
        let task = sample_task(
            "Chatten",
            "Erstelle eine C++-Konsolenanwendung mit persistenter Datenspeicherung, threadsicherer Nachrichtenverarbeitung und Build-Verifikation.",
        );
        let mut raw_inclusions = Vec::new();
        append_owner_operation_inclusions(&paths, &task, &mut raw_inclusions);
        assert!(raw_inclusions.iter().any(|item| {
            item.source_kind == "repo_operation_skill"
                && item.content.contains("Workspace Execution Operations")
        }));
        assert!(raw_inclusions.iter().any(|item| {
            item.source_kind == "system_capability_contract"
                && item.content
                    .to_ascii_lowercase()
                    .contains("repo-local workspace")
        }));
    }

    #[test]
    fn owner_interrupt_always_includes_codex_command_exec_skill_and_contract() {
        let root = temp_root("codex_exec_inclusions");
        std::fs::write(
            root.join(".agents/skills/codex-command-exec-operations/SKILL.md"),
            "# Codex Command Exec Operations\nUse this skill for command_exec execSessionAction and execCommand lifecycle handling.\n",
        )
        .unwrap();
        std::fs::write(
            root.join("contracts/system/codex-command-exec-capability-policy.json"),
            "{\"version\":1,\"purpose\":\"codex command_exec session lifecycle\"}",
        )
        .unwrap();
        let paths = sample_paths(&root);
        let task = sample_task(
            "Build the C++ console app",
            "Continue the repo-local implementation with exec sessions and bounded verification.",
        );
        let mut raw_inclusions = Vec::new();
        append_owner_operation_inclusions(&paths, &task, &mut raw_inclusions);
        assert!(raw_inclusions.iter().any(|item| {
            item.source_kind == "repo_operation_skill"
                && item
                    .source_ref
                    .ends_with("codex-command-exec-operations/SKILL.md")
        }));
        assert!(raw_inclusions.iter().any(|item| {
            item.source_kind == "system_capability_contract"
                && item
                    .source_ref
                    .ends_with("codex-command-exec-capability-policy.json")
        }));
    }

    #[test]
    fn select_exec_sessions_filters_out_foreign_task_sessions() {
        let entries = vec![
            crate::command_exec::SessionSnapshot {
                session_id: "task-37-cpp-impl-r2".to_string(),
                created_at: "2026-03-20T22:20:00Z".to_string(),
                status: "active".to_string(),
                cwd: "/home/metricspace/cto-agent".to_string(),
                tty: false,
                stream_stdin: true,
                stream_stdout_stderr: false,
                output_bytes_cap: Some(32768),
                command: vec!["bash".to_string()],
                exit_code: None,
                stdout: "src/main.cpp".to_string(),
                stderr: String::new(),
            },
            crate::command_exec::SessionSnapshot {
                session_id: "task-95-mail-reply".to_string(),
                created_at: "2026-03-20T22:21:00Z".to_string(),
                status: "active".to_string(),
                cwd: "/home/metricspace/cto-agent".to_string(),
                tty: false,
                stream_stdin: true,
                stream_stdout_stderr: false,
                output_bytes_cap: Some(32768),
                command: vec!["bash".to_string()],
                exit_code: None,
                stdout: "mail thread".to_string(),
                stderr: String::new(),
            },
        ];

        let sessions = select_exec_sessions(entries, 95, 6);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, "task-95-mail-reply");
    }

    #[test]
    fn current_task_machine_evidence_inclusion_captures_verified_workspace_anchor() {
        let root = temp_root("machine_anchor");
        let paths = sample_paths(&root);
        crate::runtime_db::init_runtime_db(&paths).expect("runtime schema should initialize");
        crate::runtime_db::enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Continue workspace implementation",
            "Use the verified workspace anchor and continue the repo task.",
            1000,
        )
        .expect("task should enqueue");
        crate::runtime_db::record_task_checkpoint(
            &paths,
            1,
            "continue",
            "Executed single bounded command successfully.",
            "Bounded exec result:\nCommand: [\"bash\", \"-lc\", \"find src -maxdepth 2 -name '*.cpp'\"]\nWorkdir: /workspace/chat-app\nSTDOUT:\nPWD=/workspace/chat-app\nsrc/main.cpp\nCMakeLists.txt\n\nSTDERR:\n",
        )
        .expect("checkpoint should persist");
        let checkpoints = crate::runtime_db::list_task_checkpoints(&paths, 1, 2)
            .expect("checkpoints should load");
        let mut raw_inclusions = Vec::new();
        append_current_task_machine_evidence_inclusion(None, 1, &checkpoints, &mut raw_inclusions);
        let anchor = raw_inclusions
            .iter()
            .find(|item| item.source_kind == "current_task_machine_evidence")
            .expect("machine anchor should be present");
        assert!(anchor.content.contains("/workspace/chat-app"));
        assert!(anchor.content.contains("src/main.cpp"));
        assert!(anchor.content.contains("CMakeLists.txt"));
    }

    #[test]
    fn current_task_machine_evidence_inclusion_uses_summary_markers_when_detail_format_shifts() {
        let checkpoints = vec![TaskCheckpointRecord {
            task_id: 7,
            created_at: "2026-03-20T18:00:00Z".to_string(),
            checkpoint_kind: "continue".to_string(),
            summary:
                "Executed single bounded command successfully. Bounded command-exec executed: [\"bash\", \"-lc\", \"git status --short && sed -n '1,200p' src/main.cpp\"]".to_string(),
            detail:
                "Command: [\"bash\", \"-lc\", \"git status --short && sed -n '1,200p' src/main.cpp\"]\nWorkdir: /workspace/chat-app\nSTDOUT:\nM src/main.cpp\nsrc/main.cpp\n".to_string(),
        }];
        let mut raw_inclusions = Vec::new();
        append_current_task_machine_evidence_inclusion(None, 7, &checkpoints, &mut raw_inclusions);
        let anchor = raw_inclusions
            .iter()
            .find(|item| item.source_kind == "current_task_machine_evidence")
            .expect("machine anchor should be present from summary markers too");
        assert!(anchor.content.contains("/workspace/chat-app"));
        assert!(anchor.content.contains("src/main.cpp"));
    }

    #[test]
    fn current_task_machine_evidence_skips_failed_exec_session_checkpoints() {
        let checkpoints = vec![
            TaskCheckpointRecord {
                task_id: 37,
                created_at: "2026-03-20T19:44:50Z".to_string(),
                checkpoint_kind: "continue".to_string(),
                summary: "Exec-session action failed.".to_string(),
                detail: "Exec session error: no active exec session found: task-37-turn-365"
                    .to_string(),
            },
            TaskCheckpointRecord {
                task_id: 37,
                created_at: "2026-03-20T19:44:42Z".to_string(),
                checkpoint_kind: "continue".to_string(),
                summary: "Read Codex exec session task-37-turn-365.".to_string(),
                detail: "Exec session result:\nSession: task-37-turn-365\nRequested command: [\"bash\"]\nCWD: /home/metricspace/cto-agent\n--- stdout ---\n/home/metricspace/cto-agent\nsrc/main.cpp\nCMakeLists.txt\n--- stderr ---\n".to_string(),
            },
        ];
        let mut raw_inclusions = Vec::new();
        append_current_task_machine_evidence_inclusion(None, 37, &checkpoints, &mut raw_inclusions);
        let anchor = raw_inclusions
            .iter()
            .find(|item| item.source_kind == "current_task_machine_evidence")
            .expect("machine anchor should prefer verified session output");
        assert!(
            anchor
                .content
                .contains("Read Codex exec session task-37-turn-365")
        );
        assert!(
            anchor
                .content
                .contains("Workdir: /home/metricspace/cto-agent")
        );
        assert!(anchor.content.contains("Command: [\"bash\"]"));
        assert!(anchor.content.contains("src/main.cpp"));
    }

    #[test]
    fn current_task_machine_evidence_survives_newer_planning_only_checkpoints() {
        let checkpoints = vec![
            TaskCheckpointRecord {
                task_id: 37,
                created_at: "2026-03-20T20:34:19Z".to_string(),
                checkpoint_kind: "continue".to_string(),
                summary:
                    "Workspace execution contract is active and no machine path ran in this turn, so persisted progress stays at planning or inspection only."
                        .to_string(),
                detail: "Planning-only continuation without a machine step.".to_string(),
            },
            TaskCheckpointRecord {
                task_id: 37,
                created_at: "2026-03-20T20:34:11Z".to_string(),
                checkpoint_kind: "continue".to_string(),
                summary:
                    "Workspace execution contract is active and no machine path ran in this turn, so persisted progress stays at planning or inspection only."
                        .to_string(),
                detail: "Another planning-only continuation without a machine step.".to_string(),
            },
            TaskCheckpointRecord {
                task_id: 37,
                created_at: "2026-03-20T19:44:42Z".to_string(),
                checkpoint_kind: "continue".to_string(),
                summary: "Read Codex exec session task-37-turn-365.".to_string(),
                detail: "Exec session result:\nSession: task-37-turn-365\nRequested command: [\"bash\"]\nCWD: /home/metricspace/cto-agent\n--- stdout ---\n/home/metricspace/cto-agent\nsrc/main.cpp\nCMakeLists.txt\n--- stderr ---\n".to_string(),
            },
        ];
        let mut raw_inclusions = Vec::new();
        append_current_task_machine_evidence_inclusion(None, 37, &checkpoints, &mut raw_inclusions);
        let anchor = raw_inclusions
            .iter()
            .find(|item| item.source_kind == "current_task_machine_evidence")
            .expect("older verified machine anchor should still be present");
        assert!(anchor.content.contains("task-37-turn-365"));
        assert!(anchor.content.contains("src/main.cpp"));
        assert!(!anchor.content.contains("planning-only continuation"));
    }

    #[test]
    fn current_task_machine_evidence_inclusion_falls_back_to_persisted_history() {
        let root = temp_root("machine_anchor_history_fallback");
        let paths = sample_paths(&root);
        crate::runtime_db::init_runtime_db(&paths).expect("runtime schema should initialize");
        crate::runtime_db::enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Continue workspace implementation",
            "Use the verified workspace anchor and continue the repo task.",
            1000,
        )
        .expect("task should enqueue");
        crate::runtime_db::record_task_checkpoint(
            &paths,
            1,
            "continue",
            "Read Codex exec session task-1-turn-11.",
            "Exec session result:\nSession: task-1-turn-11\nRequested command: [\"bash\"]\nCWD: /workspace/chat-app\n--- stdout ---\n/workspace/chat-app\nsrc/main.cpp\nCMakeLists.txt\n--- stderr ---\n",
        )
        .expect("machine checkpoint should persist");
        crate::runtime_db::record_task_checkpoint(
            &paths,
            1,
            "continue",
            "Workspace execution contract is active and no machine path ran in this turn, so persisted progress stays at planning or inspection only.",
            "Planning-only continuation without a machine step.",
        )
        .expect("planning checkpoint should persist");
        crate::runtime_db::record_task_checkpoint(
            &paths,
            1,
            "continue",
            "Workspace execution contract is active and no machine path ran in this turn, so persisted progress stays at planning or inspection only.",
            "Another planning-only continuation without a machine step.",
        )
        .expect("planning checkpoint should persist");
        let checkpoints = crate::runtime_db::list_task_checkpoints(&paths, 1, 2)
            .expect("checkpoints should load");
        assert!(
            checkpoints
                .iter()
                .all(|entry| !checkpoint_contains_machine_evidence(&entry.summary, &entry.detail))
        );
        let mut raw_inclusions = Vec::new();
        append_current_task_machine_evidence_inclusion(
            Some(&paths),
            1,
            &checkpoints,
            &mut raw_inclusions,
        );
        let anchor = raw_inclusions
            .iter()
            .find(|item| item.source_kind == "current_task_machine_evidence")
            .expect("persisted machine anchor should be recovered from history");
        assert!(anchor.content.contains("/workspace/chat-app"));
        assert!(anchor.content.contains("src/main.cpp"));
        assert!(anchor.content.contains("CMakeLists.txt"));
    }

    #[test]
    fn non_preparation_context_looks_back_far_enough_to_keep_machine_anchors_alive() {
        let policy = crate::contracts::ContextPolicy {
            version: 1,
            default_mode: "execution".to_string(),
            system_channels: vec![],
            forensic_task_kinds: vec![],
            minimal_task_kinds: vec![],
            modes: vec![
                ContextModePolicy {
                    mode: "execution".to_string(),
                    budget_hint: 8192,
                    recent_boot_entries: 0,
                    recent_bios_dialogue: 0,
                    recent_memory_items: 0,
                    recent_task_checkpoints: 1,
                    include_owner_summary: false,
                    include_raw_task_detail: true,
                },
                ContextModePolicy {
                    mode: "forensic".to_string(),
                    budget_hint: 65536,
                    recent_boot_entries: 0,
                    recent_bios_dialogue: 0,
                    recent_memory_items: 0,
                    recent_task_checkpoints: 3,
                    include_owner_summary: false,
                    include_raw_task_detail: true,
                },
            ],
            updated_at: "2026-03-20T00:00:00Z".to_string(),
        };
        let base_mode = policy.modes[0].clone();
        assert_eq!(
            checkpoint_lookback_limit(&policy, &base_mode, "owner_interrupt"),
            4
        );
        assert_eq!(
            checkpoint_lookback_limit(&policy, &base_mode, "context_preparation"),
            3
        );
    }

    #[test]
    fn sticky_execute_mode_remains_in_allowed_next_modes() {
        let policy = ModeSystemPolicy {
            version: 1,
            initial_mode: "execute_task".to_string(),
            preferred_operating_goal: "finish_current_task_with_verified_progress".to_string(),
            modes: vec![
                "execute_task".to_string(),
                "review".to_string(),
                "blocked".to_string(),
            ],
            transitions: vec![
                crate::contracts::ModeTransitionRule {
                    from_mode: "execute_task".to_string(),
                    to_mode: "review".to_string(),
                    rationale: "after bounded step".to_string(),
                },
                crate::contracts::ModeTransitionRule {
                    from_mode: "execute_task".to_string(),
                    to_mode: "blocked".to_string(),
                    rationale: "on hard failure".to_string(),
                },
            ],
            updated_at: "2026-03-20T00:00:00Z".to_string(),
        };
        let modes = allowed_next_modes(&policy, "execute_task");
        assert_eq!(modes.first().map(String::as_str), Some("execute_task"));
        assert!(modes.iter().any(|mode| mode == "review"));
        assert!(modes.iter().any(|mode| mode == "blocked"));
    }

    #[test]
    fn task_definition_of_done_policy_is_included_when_present() {
        let root = temp_root("dod_policy");
        std::fs::write(
            root.join("contracts/system/task-definition-of-done-policy.json"),
            "{\"version\":1,\"purpose\":\"done\"}",
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
            installation_bootstrap_path: root
                .join("contracts/bootstrap/installation-bootstrap.json"),
            context_policy_path: root.join("contracts/context/context-policy.json"),
            context_optimization_policy_path: root
                .join("contracts/context/context-optimization-policy.json"),
            context_query_tool_contract_path: root
                .join("contracts/context/context-query-tool-contract.json"),
            context_governance_policy_path: root
                .join("contracts/context/context-governance-policy.json"),
            mode_system_policy_path: root.join("contracts/system/mode-system-policy.json"),
            loop_safety_policy_path: root.join("contracts/system/loop-safety-policy.json"),
            execution_authority_policy_path: root
                .join("contracts/system/execution-authority-policy.json"),
            browser_engine_policy_path: root.join("contracts/browser/browser-engine-policy.json"),
            browser_capability_policy_path: root
                .join("contracts/browser/browser-capability-policy.json"),
            browser_subworker_policy_path: root
                .join("contracts/browser/browser-subworker-policy.json"),
            self_preservation_state_path: root
                .join("contracts/system/self-preservation-state.json"),
            origin_story_path: root.join("contracts/history/origin-story.md"),
            creation_ledger_path: root.join("contracts/history/creation-ledger.md"),
            boot_log_path: root.join("runtime/boot_log.jsonl"),
            agent_state_path: root.join("runtime/state/agent_state.json"),
            system_census_path: root.join("runtime/state/system_census.json"),
            browser_engine_state_path: root.join("runtime/state/browser_engine_state.json"),
            runtime_db_path: root.join("runtime/cto_agent.db"),
            attach_socket_path: root.join("runtime/cto-agent.sock"),
            runtime_lock_path: root.join("runtime/cto-agent.lock"),
            pending_hard_reset_report_path: root
                .join("runtime/recovery/pending-hard-reset-report.json"),
            certs_dir: root.join("runtime/certs"),
            tls_cert_path: root.join("runtime/certs/localhost.crt"),
            tls_key_path: root.join("runtime/certs/localhost.key"),
        };
        let mut raw_inclusions = Vec::new();
        append_definition_of_done_inclusions(&paths, &mut raw_inclusions);
        assert_eq!(raw_inclusions.len(), 1);
        assert_eq!(
            raw_inclusions[0].source_kind,
            "task_definition_of_done_policy"
        );
        assert!(raw_inclusions[0].content.contains("done"));
    }

    #[test]
    fn installation_tasks_include_tool_smoke_resource() {
        let root = temp_root("install_smoke");
        std::fs::create_dir_all(root.join("contracts/bootstrap")).unwrap();
        std::fs::write(
            root.join("contracts/bootstrap/installation-tool-smoke-resource.json"),
            "{\"version\":1,\"purpose\":\"tool-smokes\"}",
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
            installation_bootstrap_path: root
                .join("contracts/bootstrap/installation-bootstrap.json"),
            context_policy_path: root.join("contracts/context/context-policy.json"),
            context_optimization_policy_path: root
                .join("contracts/context/context-optimization-policy.json"),
            context_query_tool_contract_path: root
                .join("contracts/context/context-query-tool-contract.json"),
            context_governance_policy_path: root
                .join("contracts/context/context-governance-policy.json"),
            mode_system_policy_path: root.join("contracts/system/mode-system-policy.json"),
            loop_safety_policy_path: root.join("contracts/system/loop-safety-policy.json"),
            execution_authority_policy_path: root
                .join("contracts/system/execution-authority-policy.json"),
            browser_engine_policy_path: root.join("contracts/browser/browser-engine-policy.json"),
            browser_capability_policy_path: root
                .join("contracts/browser/browser-capability-policy.json"),
            browser_subworker_policy_path: root
                .join("contracts/browser/browser-subworker-policy.json"),
            self_preservation_state_path: root
                .join("contracts/system/self-preservation-state.json"),
            origin_story_path: root.join("contracts/history/origin-story.md"),
            creation_ledger_path: root.join("contracts/history/creation-ledger.md"),
            boot_log_path: root.join("runtime/boot_log.jsonl"),
            agent_state_path: root.join("runtime/state/agent_state.json"),
            system_census_path: root.join("runtime/state/system_census.json"),
            browser_engine_state_path: root.join("runtime/state/browser_engine_state.json"),
            runtime_db_path: root.join("runtime/cto_agent.db"),
            attach_socket_path: root.join("runtime/cto-agent.sock"),
            runtime_lock_path: root.join("runtime/cto-agent.lock"),
            pending_hard_reset_report_path: root
                .join("runtime/recovery/pending-hard-reset-report.json"),
            certs_dir: root.join("runtime/certs"),
            tls_cert_path: root.join("runtime/certs/localhost.crt"),
            tls_key_path: root.join("runtime/certs/localhost.key"),
        };
        let task = TaskRecord {
            task_kind: "bootstrap_runtime_guard".to_string(),
            ..sample_task("Bootstrap tool verification", "Smoke the real tools.")
        };
        let mut raw_inclusions = Vec::new();
        append_installation_tool_smoke_inclusions(&paths, &task, &mut raw_inclusions);
        assert_eq!(raw_inclusions.len(), 1);
        assert_eq!(
            raw_inclusions[0].source_kind,
            "installation_tool_smoke_resource"
        );
        assert!(raw_inclusions[0].content.contains("tool-smokes"));
    }

    #[test]
    fn prepared_context_artifact_is_extracted_from_checkpoint_text() {
        let artifact = extract_prepared_context_artifact_from_text(
            r#"Preparation artifact:
            {
              "preparedContextArtifact": {
                "immediateNextStep": "Inspect the C++ project root.",
                "questions": [
                  {
                    "question": "Which repo path is verified?",
                    "why": "The next direct step depends on the true workspace root."
                  }
                ],
                "blocks": [
                  {
                    "blockId": "goal_and_authority",
                    "title": "Goal And Authority",
                    "tokenBudget": 180,
                    "content": "The owner requested implementation of the C++ console app.",
                    "whyIncluded": "This is the objective.",
                    "evidenceRefs": ["task:22"]
                  }
                ],
                "review": {
                  "decision": "revise",
                  "note": "Need a verified world-state block.",
                  "missingEvidence": ["Current repo path"],
                  "weakBlocks": ["verified_world_state"]
                }
              }
            }"#,
        )
        .expect("artifact should parse");

        assert_eq!(artifact.questions.len(), 1);
        assert_eq!(artifact.blocks[0].block_id, "goal_and_authority");
        assert_eq!(artifact.review.decision, "revise");
    }

    #[test]
    fn context_query_answers_rank_task_detail_for_matching_question() {
        let root = temp_root("context_query_answers");
        let paths = sample_paths(&root);
        let task = TaskRecord {
            task_kind: "context_preparation".to_string(),
            title: "Prepare context for C++ app".to_string(),
            detail: "Build the C++ console app with registration, friends, messages, persistence, and thread safety.".to_string(),
            ..sample_task("Prepare context", "Build the C++ app.")
        };
        let artifact = ContextPreparedArtifact {
            immediate_next_step: "Verify the active workspace before writing the first module.".to_string(),
            questions: vec![ContextQueryQuestion {
                question_id: "active_cpp_task".to_string(),
                question: "Which task is about the C++ console app with registration and thread-safe messaging?".to_string(),
                why: "The handoff should stay pinned to the actual implementation task.".to_string(),
                query_mode: "sqlite_hybrid".to_string(),
                source_kinds: vec!["task_detail".to_string()],
                max_matches: Some(2),
                required_keywords: vec!["c++".to_string(), "thread-safe".to_string()],
            }],
            blocks: Vec::new(),
            review: ContextPreparedReview {
                decision: "query_more".to_string(),
                note: "Need the best matching current task evidence.".to_string(),
                missing_evidence: vec!["Verified active task evidence".to_string()],
                weak_blocks: vec!["verified_world_state".to_string()],
                budget_violations: Vec::new(),
                repeated_from_prior: false,
                findings: Vec::new(),
                assessment: None,
            },
        };

        let answers = build_context_query_answers(
            &paths,
            &task,
            &crate::contracts::default_context_optimization_policy(),
            &crate::contracts::default_context_query_tool_contract(),
            Some(&artifact),
        );
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].matches[0].source_kind, "task_detail");
        assert!(answers[0].matches[0].excerpt.contains("thread safety"));
    }

    #[test]
    fn context_query_answers_honor_requested_query_mode() {
        let root = temp_root("context_query_modes");
        let paths = sample_paths(&root);
        let task = TaskRecord {
            task_kind: "context_preparation".to_string(),
            title: "Prepare context for browser repair".to_string(),
            detail: "Inspect the browser capability and desktop session bridge before repair."
                .to_string(),
            ..sample_task("Prepare context", "Inspect browser repair path.")
        };
        let artifact = ContextPreparedArtifact {
            immediate_next_step: "Locate the strongest browser-bridge evidence.".to_string(),
            questions: vec![
                ContextQueryQuestion {
                    question_id: "browser_bridge_fts".to_string(),
                    question:
                        "Which evidence talks about the browser capability and desktop session bridge?"
                            .to_string(),
                    why: "The preparation loop should use the requested retrieval mode."
                        .to_string(),
                    query_mode: "sqlite_fts".to_string(),
                    source_kinds: vec!["task_detail".to_string()],
                    max_matches: Some(2),
                    required_keywords: vec!["browser".to_string(), "desktop session".to_string()],
                },
                ContextQueryQuestion {
                    question_id: "browser_bridge_hybrid".to_string(),
                    question:
                        "Which evidence best grounds the browser capability and desktop session repair step?"
                            .to_string(),
                    why: "Hybrid mode should be marked as hybrid in the resulting matches."
                        .to_string(),
                    query_mode: "sqlite_hybrid".to_string(),
                    source_kinds: vec!["task_detail".to_string()],
                    max_matches: Some(2),
                    required_keywords: vec!["browser".to_string(), "desktop session".to_string()],
                },
            ],
            blocks: Vec::new(),
            review: ContextPreparedReview {
                decision: "query_more".to_string(),
                note: "Need grounded retrieval first.".to_string(),
                missing_evidence: vec!["Relevant browser bridge evidence".to_string()],
                weak_blocks: vec!["verified_world_state".to_string()],
                budget_violations: Vec::new(),
                repeated_from_prior: false,
                findings: Vec::new(),
                assessment: None,
            },
        };

        let answers = build_context_query_answers(
            &paths,
            &task,
            &crate::contracts::default_context_optimization_policy(),
            &crate::contracts::default_context_query_tool_contract(),
            Some(&artifact),
        );
        assert_eq!(answers.len(), 2);
        assert!(
            answers[0].matches[0]
                .query_mode_used
                .starts_with("sqlite_fts")
        );
        assert_eq!(answers[1].matches[0].query_mode_used, "sqlite_hybrid");
    }

    #[test]
    fn context_query_answers_can_use_current_task_checkpoint_evidence() {
        let root = temp_root("context_query_task_checkpoint");
        let paths = sample_paths(&root);
        crate::runtime_db::init_runtime_db(&paths).expect("runtime schema should initialize");
        let task = TaskRecord {
            id: 42,
            task_kind: "context_preparation".to_string(),
            title: "Prepare context for C++ app".to_string(),
            detail: "Build the C++ console app with persistence and thread-safe messaging."
                .to_string(),
            ..sample_task("Prepare context", "Build the C++ app.")
        };
        crate::runtime_db::record_task_checkpoint(
            &paths,
            task.id,
            "context_query_plan",
            "Repo scan captured the verified workspace state.",
            "Verified repo root: /workspace/chat-app\nBuild files: CMakeLists.txt\nSources: src/main.cpp",
        )
        .expect("checkpoint should persist");
        let artifact = ContextPreparedArtifact {
            immediate_next_step: "Rewrite the execution handoff around the verified repo scan."
                .to_string(),
            questions: vec![ContextQueryQuestion {
                question_id: "verified_repo_scan".to_string(),
                question: "Which evidence already shows the verified repo root and CMake build files for the C++ app?".to_string(),
                why: "The rewrite phase should reuse the bounded repo scan instead of asking again.".to_string(),
                query_mode: "sqlite_hybrid".to_string(),
                source_kinds: vec!["task_checkpoint".to_string()],
                max_matches: Some(2),
                required_keywords: vec!["cmakelists".to_string(), "repo".to_string()],
            }],
            blocks: Vec::new(),
            review: ContextPreparedReview {
                decision: "query_more".to_string(),
                note: "Need the current checkpoint evidence.".to_string(),
                missing_evidence: vec!["Verified repo scan".to_string()],
                weak_blocks: vec!["verified_world_state".to_string()],
                budget_violations: Vec::new(),
                repeated_from_prior: false,
                findings: Vec::new(),
                assessment: None,
            },
        };

        let answers = build_context_query_answers(
            &paths,
            &task,
            &crate::contracts::default_context_optimization_policy(),
            &crate::contracts::default_context_query_tool_contract(),
            Some(&artifact),
        );
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].matches[0].source_kind, "task_checkpoint");
        assert!(answers[0].matches[0].excerpt.contains("CMakeLists"));
    }

    #[test]
    fn context_preparation_advances_to_rewrite_after_query_answers_exist() {
        let task = TaskRecord {
            task_kind: "context_preparation".to_string(),
            ..sample_task("Prepare context", "Build the C++ console app.")
        };
        let artifact = ContextPreparedArtifact {
            immediate_next_step: "Ask the context store for the strongest repo-grounding facts."
                .to_string(),
            questions: vec![ContextQueryQuestion {
                question_id: "repo_root".to_string(),
                question: "Which verified workspace root contains the C++ app task?".to_string(),
                why: "The rewrite phase needs a grounded repo path.".to_string(),
                query_mode: "sqlite_hybrid".to_string(),
                source_kinds: vec!["task_detail".to_string()],
                max_matches: Some(2),
                required_keywords: vec!["c++".to_string(), "workspace".to_string()],
            }],
            blocks: Vec::new(),
            review: ContextPreparedReview {
                decision: "query_more".to_string(),
                note: "Need the strongest retrieval evidence before rewriting the handoff."
                    .to_string(),
                missing_evidence: vec!["Verified workspace root".to_string()],
                weak_blocks: vec!["verified_world_state".to_string()],
                budget_violations: Vec::new(),
                repeated_from_prior: false,
                findings: Vec::new(),
                assessment: None,
            },
        };
        let answers = vec![ContextQueryAnswer {
            question: "Which verified workspace root contains the C++ app task?".to_string(),
            why: "The rewrite phase needs a grounded repo path.".to_string(),
            matches: vec![ContextQueryMatch {
                source_kind: "task_detail".to_string(),
                source_ref: "task:25".to_string(),
                summary: "Task 25".to_string(),
                excerpt: "Build the C++ console app in the active workspace.".to_string(),
                score: 100,
                query_mode_used: "sqlite_hybrid".to_string(),
            }],
        }];

        assert_eq!(
            infer_context_optimization_phase(&task, Some(&artifact), &answers),
            "rewrite"
        );
    }

    #[test]
    fn context_preparation_phase_mode_override_is_used() {
        let policy = crate::contracts::default_context_policy();
        let optimization = crate::contracts::default_context_optimization_policy();
        let task = TaskRecord {
            task_kind: "context_preparation".to_string(),
            ..sample_task("Prepare context", "Build the C++ console app.")
        };

        let query_mode = choose_mode(&policy, &optimization, &task, Some("query_plan"));
        assert_eq!(query_mode.mode, "preparation_query");
        assert!(query_mode.budget_hint >= 16_384);

        let rewrite_mode = choose_mode(&policy, &optimization, &task, Some("rewrite"));
        assert_eq!(rewrite_mode.mode, "preparation_rewrite");

        let review_mode = choose_mode(&policy, &optimization, &task, Some("review"));
        assert_eq!(review_mode.mode, "preparation_review");
    }

    #[test]
    fn prepare_context_package_emits_context_distillation_from_prepared_artifact() {
        let root = temp_root("context_distillation_prepared_artifact");
        let paths = sample_paths(&root);
        crate::runtime_db::init_runtime_db(&paths).expect("runtime schema should initialize");
        let task = TaskRecord {
            id: 77,
            task_kind: "workspace_repair".to_string(),
            title: "Repair browser bridge".to_string(),
            detail: "Repair the browser bridge and verify the desktop session path.".to_string(),
            ..sample_task("Repair browser bridge", "Repair the browser bridge.")
        };
        crate::runtime_db::record_task_checkpoint(
            &paths,
            task.id,
            "context_preparation_ready",
            "Prepared execution handoff is ready.",
            r#"{
              "preparedContextArtifact": {
                "immediateNextStep": "Run the desktop bridge verification command.",
                "blocks": [
                  {
                    "blockId": "goal_and_authority",
                    "title": "Goal And Authority",
                    "tokenBudget": 180,
                    "content": "Repair the browser bridge for the owner and keep the desktop session path intact.",
                    "whyIncluded": "Sets the task authority.",
                    "evidenceRefs": ["task:77"]
                  },
                  {
                    "blockId": "definition_of_done",
                    "title": "Definition Of Done",
                    "tokenBudget": 180,
                    "content": "The browser bridge works again and the desktop session verification passes with fresh evidence.",
                    "whyIncluded": "Defines real completion.",
                    "evidenceRefs": ["task:77"]
                  },
                  {
                    "blockId": "verified_world_state",
                    "title": "Verified World State",
                    "tokenBudget": 180,
                    "content": "The extension code exists, but the current desktop bridge runtime still needs a fresh verification run.",
                    "whyIncluded": "Captures the current verified state.",
                    "evidenceRefs": ["task:77:context_preparation_ready"]
                  },
                  {
                    "blockId": "next_action_only",
                    "title": "Next Action Only",
                    "tokenBudget": 180,
                    "content": "Run the bridge verification command and capture the exact failure or success evidence.",
                    "whyIncluded": "Pins the next bounded step.",
                    "evidenceRefs": ["task:77:context_preparation_ready"]
                  }
                ],
                "review": {
                  "decision": "go",
                  "note": "The prepared handoff is grounded enough for execution.",
                  "missingEvidence": [],
                  "weakBlocks": [],
                  "budgetViolations": []
                }
              }
            }"#,
        )
        .expect("checkpoint should persist");

        let package = prepare_context_package(&paths, &task).expect("package should build");
        let distillation = package
            .context_distillation
            .expect("distillation should be present");
        assert!(
            distillation
                .continuity_narrative
                .contains("desktop bridge runtime")
        );
        assert!(
            distillation
                .continuity_artifacts
                .iter()
                .any(|artifact| artifact.kind == "prepared_block")
        );
        assert_eq!(
            distillation.active_focus.next_step,
            "Run the desktop bridge verification command."
        );
        assert!(
            distillation
                .continuity_anchors
                .iter()
                .any(|anchor| anchor.anchor_id == "definition_of_done")
        );
        assert!(
            distillation
                .historical_retrieval_refs
                .iter()
                .any(|item| item.source_kind == "context_distillation_focus")
        );
    }

    #[test]
    fn context_query_candidates_include_saved_context_distillation_sources() {
        let root = temp_root("context_distillation_query_candidates");
        let paths = sample_paths(&root);
        crate::runtime_db::init_runtime_db(&paths).expect("runtime schema should initialize");
        let task = TaskRecord {
            id: 91,
            task_kind: "progress_reflection".to_string(),
            title: "Reflect on browser rollout".to_string(),
            detail: "Review the browser rollout and choose the next bounded task.".to_string(),
            ..sample_task("Reflect on browser rollout", "Review the browser rollout.")
        };
        let package_json = serde_json::json!({
            "contextDistillation": {
                "continuityNarrative": "The browser rollout already stabilized the extension path, but the next task must verify the desktop bridge before wider deployment.",
                "continuityArtifacts": [
                    {
                        "artifactId": "artifact_1",
                        "kind": "checkpoint",
                        "label": "Latest Checkpoint",
                        "summary": "Desktop bridge verification remains open.",
                        "sourceRef": "task:91:continue"
                    }
                ],
                "continuityAnchors": [
                    {
                        "anchorId": "rollout_state",
                        "title": "Rollout State",
                        "content": "Extension path stabilized; desktop bridge still needs a final bounded verification.",
                        "evidenceRefs": ["task:91"]
                    }
                ],
                "systemContinuityAnchors": [
                    {
                        "anchorId": "priority_and_authority",
                        "title": "Priority And Authority",
                        "content": "Owner priority stays above self-extension.",
                        "evidenceRefs": ["distillation:system:91:priority_and_authority"]
                    }
                ],
                "activeFocus": {
                    "status": "Desktop bridge verification is still open.",
                    "blocker": "",
                    "nextStep": "Verify the desktop bridge.",
                    "doneCriteria": "Fresh bridge evidence exists.",
                    "evidenceRefs": ["task:91"]
                },
                "snapshot": {
                    "summary": "Browser rollout reflection is active.",
                    "bulletPoints": ["Context mode: compact"],
                    "evidenceRefs": ["task:91"]
                },
                "historicalRetrievalRefs": [
                    {
                        "sourceKind": "task_detail",
                        "sourceRef": "task:91",
                        "label": "Task #91"
                    }
                ]
            }
        })
        .to_string();
        crate::runtime_db::record_context_package(
            &paths,
            task.id,
            &task.title,
            "compact",
            1600,
            "test distillation",
            &package_json,
        )
        .expect("context package should persist");

        let candidates = build_context_query_candidates(&paths, &task);
        assert!(
            candidates
                .iter()
                .any(|item| item.source_kind == "context_distillation_focus")
        );
        assert!(
            candidates
                .iter()
                .any(|item| item.source_kind == "context_distillation_story")
        );
        assert!(
            candidates
                .iter()
                .any(|item| item.source_kind == "context_distillation_system_anchor")
        );
        assert!(
            candidates
                .iter()
                .any(|item| item.source_kind == "context_distillation_artifact")
        );
    }

    #[test]
    fn context_embedding_target_resolves_from_runtime_env_file() {
        let root = temp_root("context_embedding_target");
        let paths = sample_paths(&root);
        std::fs::create_dir_all(root.join("runtime")).unwrap();
        std::fs::write(
            root.join("runtime/kleinhirn.env"),
            "CTO_AGENT_CONTEXT_EMBEDDING_ENABLED='1'\nCTO_AGENT_CONTEXT_EMBEDDING_RUNTIME_MODEL='Qwen/Qwen3-Embedding-0.6B'\nCTO_AGENT_CONTEXT_EMBEDDING_PORT='1235'\nCTO_AGENT_CONTEXT_EMBEDDING_BASE_URL='http://127.0.0.1:1235/v1'\nCTO_AGENT_CONTEXT_EMBEDDING_API_KEY='local-context-embedding'\n",
        )
        .unwrap();

        let target = resolve_context_embedding_target(&paths).expect("embedding target");
        assert_eq!(target.model_id, "Qwen/Qwen3-Embedding-0.6B");
        assert_eq!(target.base_url, "http://127.0.0.1:1235/v1");
        assert_eq!(target.max_batch_size, 12);
    }

    #[test]
    fn context_query_contract_summary_reflects_runtime_embedding_config() {
        let root = temp_root("context_query_contract_summary");
        let paths = sample_paths(&root);
        std::fs::create_dir_all(root.join("runtime")).unwrap();
        std::fs::write(
            root.join("runtime/kleinhirn.env"),
            "CTO_AGENT_CONTEXT_EMBEDDING_ENABLED='1'\nCTO_AGENT_CONTEXT_EMBEDDING_RUNTIME_MODEL='Qwen/Qwen3-Embedding-0.6B'\nCTO_AGENT_CONTEXT_EMBEDDING_BASE_URL='http://127.0.0.1:1235/v1'\n",
        )
        .unwrap();

        let contract = crate::contracts::default_context_query_tool_contract();
        let summary = build_context_query_contract_summary(
            &contract,
            resolve_context_embedding_target(&paths).as_ref(),
        );
        assert!(summary.embedding_search_available);
        assert!(
            summary
                .embedding_search_note
                .contains("Qwen/Qwen3-Embedding-0.6B")
        );
    }
}

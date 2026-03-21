use crate::contracts::BootstrapTaskTemplate;
use crate::contracts::HomepagePolicy;
use crate::contracts::LoopSafetyPolicy;
use crate::contracts::ModelPolicy;
use crate::contracts::Paths;
use crate::contracts::SystemCensus;
use crate::contracts::describe_kleinhirn_selection;
use crate::contracts::load_agent_state;
use crate::contracts::load_bios;
use crate::contracts::load_bootstrap_task_pack;
use crate::contracts::load_census;
use crate::contracts::load_homepage_policy;
use crate::contracts::load_loop_safety_policy;
use crate::contracts::load_organigram;
use crate::contracts::load_root_auth;
use crate::contracts::load_self_preservation_state;
use crate::contracts::now_iso;
use crate::contracts::write_self_preservation_state;
use chrono::DateTime;
use chrono::Duration as ChronoDuration;
use chrono::Utc;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use rusqlite::TransactionBehavior;
use rusqlite::params;
use serde::Deserialize;
use serde::Serialize;
use std::fs;
use std::time::Duration;

#[derive(Debug, Clone, Serialize)]
pub struct BiosDialogueEntry {
    pub created_at: String,
    pub speaker: String,
    pub message: String,
    pub used_grosshirn: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct OwnerTrustSnapshot {
    pub owner_name: String,
    pub committed_owner_name: String,
    pub owner_contact_established: bool,
    pub bios_primary_channel_confirmed: bool,
    pub superpassword_set: bool,
    pub owner_commitment_score: i64,
    pub last_owner_dialogue_at: Option<String>,
    pub calibration_notes: String,
    pub brain_access_mode: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BrainRoutingState {
    pub route_mode: String,
    pub boosted_task_id: Option<i64>,
    pub boosted_task_title: String,
    pub boost_reason: String,
    pub boost_started_at: Option<String>,
    pub boost_last_used_at: Option<String>,
    pub boost_expires_at: Option<String>,
    pub cooldown_until: Option<String>,
    pub last_deactivation_reason: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct ExternalBrainCostSummary {
    pub grosshirn_calls: i64,
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_tokens: i64,
    pub estimated_cost_usd: f64,
    pub last_external_call_at: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BrainUsageRollup {
    pub total_input_tokens: i64,
    pub total_output_tokens: i64,
    pub total_tokens: i64,
    pub last_model_id: String,
    pub last_brain_tier: String,
    pub last_input_tokens: i64,
    pub last_output_tokens: i64,
    pub last_total_tokens: i64,
    pub last_recorded_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct MemoryItemRecord {
    pub created_at: String,
    pub kind: String,
    pub summary: String,
    pub detail: String,
    pub source: String,
    pub important: bool,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct LearningEntryDraft {
    pub learning_class: String,
    pub summary: String,
    pub detail: String,
    pub evidence: String,
    pub applicability: String,
    pub confidence: f64,
    pub salience: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct LearningEntryRecord {
    pub id: i64,
    pub created_at: String,
    pub updated_at: String,
    pub source_task_id: Option<i64>,
    pub source_turn_id: Option<i64>,
    pub source_task_kind: String,
    pub source_channel: String,
    pub learning_class: String,
    pub status: String,
    pub summary: String,
    pub detail: String,
    pub evidence: String,
    pub applicability: String,
    pub confidence: f64,
    pub salience: i64,
    pub recall_count: i64,
    pub last_recalled_at: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PersonProfileRecord {
    pub id: i64,
    pub created_at: String,
    pub updated_at: String,
    pub canonical_key: String,
    pub display_name: String,
    pub primary_email: String,
    pub relationship_kind: String,
    pub trust_level: String,
    pub last_interaction_at: Option<String>,
    pub last_channel: String,
    pub interaction_count: i64,
    pub conversation_memory_summary: String,
    pub notebook_summary: String,
    pub proactive_guard_note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PersonNoteRecord {
    pub id: i64,
    pub person_profile_id: i64,
    pub person_display_name: String,
    pub created_at: String,
    pub updated_at: String,
    pub note_kind: String,
    pub source_channel: String,
    pub source_ref: String,
    pub summary: String,
    pub detail: String,
    pub important: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct ProactiveContactCandidateRecord {
    pub id: i64,
    pub created_at: String,
    pub updated_at: String,
    pub person_profile_id: Option<i64>,
    pub person_display_name: String,
    pub person_email: String,
    pub source_task_id: Option<i64>,
    pub source_turn_id: Option<i64>,
    pub status: String,
    pub channel: String,
    pub subject: String,
    pub draft_body: String,
    pub rationale: String,
    pub conflict_check: String,
    pub requires_grosshirn_validation: bool,
    pub validation_task_id: Option<i64>,
    pub validated_at: Option<String>,
    pub validation_decision: String,
    pub validation_note: String,
    pub dispatch_task_id: Option<i64>,
    pub dispatched_at: Option<String>,
    pub dispatch_channel: String,
    pub dispatch_note: String,
    pub outbound_message_id: String,
    pub source: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProactiveContactDraft {
    pub person_name: String,
    pub person_email: String,
    pub channel: String,
    pub subject: String,
    pub body: String,
    pub rationale: String,
    pub conflict_check: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProactiveContactValidationDraft {
    pub decision: String,
    pub note: String,
    pub revised_subject: String,
    pub revised_body: String,
}

#[derive(Debug, Clone)]
pub struct MailPreviewRecord {
    pub received_at_iso: String,
    pub direction: String,
    pub subject: String,
    pub preview: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ResourceRecord {
    pub category: String,
    pub name: String,
    pub observed_at: String,
    pub status: String,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SkillRecord {
    pub name: String,
    pub path: String,
    pub status: String,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct HomepageRevisionRecord {
    pub created_at: String,
    pub source_channel: String,
    pub title: String,
    pub headline: String,
    pub owner_branding_applied: bool,
    pub notes: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct BiosUploadRecord {
    pub created_at: String,
    pub speaker: String,
    pub source_channel: String,
    pub note: String,
    pub file_name: String,
    pub public_path: String,
    pub mime_type: String,
}

#[derive(Debug, Clone)]
pub struct LoopInterruptRecord {
    pub id: i64,
    pub created_at: String,
    pub source_channel: String,
    pub speaker: String,
    pub message: String,
    pub status: String,
    pub response: Option<String>,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct TaskRecord {
    pub id: i64,
    pub created_at: String,
    pub updated_at: String,
    pub parent_task_id: Option<i64>,
    pub worker_job_id: Option<i64>,
    pub source_interrupt_id: Option<i64>,
    pub source_channel: String,
    pub speaker: String,
    pub task_kind: String,
    pub title: String,
    pub detail: String,
    pub trust_level: String,
    pub priority_score: i64,
    pub status: String,
    pub run_count: i64,
    pub last_checkpoint_summary: Option<String>,
    pub last_checkpoint_at: Option<String>,
    pub last_output: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct FocusStateRecord {
    pub mode: String,
    pub active_task_id: Option<i64>,
    pub active_task_title: String,
    pub queue_depth: i64,
    pub last_reprioritized_at: Option<String>,
    pub last_task_completed_at: Option<String>,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct TaskCheckpointRecord {
    pub task_id: i64,
    pub created_at: String,
    pub checkpoint_kind: String,
    pub summary: String,
    pub detail: String,
}

#[derive(Debug, Clone)]
pub struct ContextPackageRecord {
    pub id: i64,
    pub created_at: String,
    pub task_id: i64,
    pub task_title: String,
    pub context_mode: String,
    pub budget_hint: i64,
    pub rationale: String,
    pub package_json: String,
}

#[derive(Debug, Clone)]
pub struct ModeTransitionRecord {
    pub created_at: String,
    pub from_mode: String,
    pub to_mode: String,
    pub active_task_id: Option<i64>,
    pub active_task_title: String,
    pub trigger: String,
    pub note: String,
}

#[derive(Debug, Clone)]
pub struct WorkerJobRecord {
    pub id: i64,
    pub created_at: String,
    pub updated_at: String,
    pub parent_task_id: i64,
    pub parent_task_title: String,
    pub worker_kind: String,
    pub contract_title: String,
    pub contract_detail: String,
    pub status: String,
    pub request_note: String,
    pub result_summary: Option<String>,
    pub result_detail: Option<String>,
    pub review_summary: Option<String>,
    pub review_detail: Option<String>,
    pub review_task_id: Option<i64>,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentEventRecord {
    pub id: i64,
    pub created_at: String,
    pub method: String,
    pub active_task_id: Option<i64>,
    pub active_task_title: String,
    pub body: String,
    pub payload_json: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentTurnRecord {
    pub id: i64,
    pub created_at: String,
    pub updated_at: String,
    pub task_id: i64,
    pub task_title: String,
    pub trigger: String,
    pub mode_at_start: String,
    pub mode_at_end: Option<String>,
    pub status: String,
    pub summary: Option<String>,
    pub output: Option<String>,
    pub completed_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct AgentThreadRecord {
    pub thread_key: String,
    pub created_at: String,
    pub updated_at: String,
    pub lifecycle_status: String,
    pub current_mode: String,
    pub active_turn_id: Option<i64>,
    pub active_task_id: Option<i64>,
    pub queue_depth: i64,
    pub note: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct TurnSignalRecord {
    pub id: i64,
    pub created_at: String,
    pub thread_key: String,
    pub turn_id: Option<i64>,
    pub task_id: Option<i64>,
    pub signal_kind: String,
    pub source_channel: String,
    pub speaker: String,
    pub message: String,
    pub status: String,
    pub consumed_at: Option<String>,
    pub resolution_note: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct LoopIncidentRecord {
    pub id: i64,
    pub created_at: String,
    pub updated_at: String,
    pub incident_key: String,
    pub severity: String,
    pub status: String,
    pub summary: String,
    pub detail: String,
    pub related_task_id: Option<i64>,
    pub related_turn_id: Option<i64>,
    pub self_preservation_task_id: Option<i64>,
    pub hard_reset_required: bool,
    pub hard_reset_report_path: Option<String>,
    pub resolved_at: Option<String>,
}

pub fn init_runtime_db(paths: &Paths) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.pragma_update(None, "journal_mode", "WAL")?;
    conn.pragma_update(None, "synchronous", "NORMAL")?;
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS bios_dialogue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            speaker TEXT NOT NULL,
            channel TEXT NOT NULL,
            message TEXT NOT NULL,
            used_grosshirn INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS owner_trust (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            owner_name TEXT NOT NULL DEFAULT '',
            committed_owner_name TEXT NOT NULL DEFAULT '',
            owner_contact_established INTEGER NOT NULL DEFAULT 0,
            bios_primary_channel_confirmed INTEGER NOT NULL DEFAULT 0,
            superpassword_set INTEGER NOT NULL DEFAULT 0,
            owner_commitment_score INTEGER NOT NULL DEFAULT 0,
            last_owner_dialogue_at TEXT,
            calibration_notes TEXT NOT NULL DEFAULT '',
            brain_access_mode TEXT NOT NULL DEFAULT 'kleinhirn_only'
        );
        CREATE TABLE IF NOT EXISTS brain_routing_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            route_mode TEXT NOT NULL DEFAULT 'kleinhirn',
            boosted_task_id INTEGER,
            boosted_task_title TEXT NOT NULL DEFAULT '',
            boost_reason TEXT NOT NULL DEFAULT '',
            boost_started_at TEXT,
            boost_last_used_at TEXT,
            boost_expires_at TEXT,
            cooldown_until TEXT,
            last_deactivation_reason TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS brain_usage_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            task_id INTEGER,
            turn_id INTEGER,
            brain_tier TEXT NOT NULL,
            source_label TEXT NOT NULL,
            model_id TEXT NOT NULL,
            input_tokens INTEGER NOT NULL DEFAULT 0,
            output_tokens INTEGER NOT NULL DEFAULT 0,
            total_tokens INTEGER NOT NULL DEFAULT 0,
            estimated_cost_usd REAL NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS memory_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            kind TEXT NOT NULL,
            summary TEXT NOT NULL,
            detail TEXT NOT NULL,
            source TEXT NOT NULL,
            important INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS memory_summaries (
            scope TEXT PRIMARY KEY,
            updated_at TEXT NOT NULL,
            summary TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS learning_entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            source_task_id INTEGER,
            source_turn_id INTEGER,
            source_task_kind TEXT NOT NULL DEFAULT '',
            source_channel TEXT NOT NULL DEFAULT '',
            learning_class TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'candidate',
            summary TEXT NOT NULL,
            detail TEXT NOT NULL,
            evidence TEXT NOT NULL,
            applicability TEXT NOT NULL DEFAULT '',
            confidence REAL NOT NULL DEFAULT 0.5,
            salience INTEGER NOT NULL DEFAULT 50,
            recall_count INTEGER NOT NULL DEFAULT 0,
            last_recalled_at TEXT,
            source TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_learning_entries_status_class
            ON learning_entries(status, learning_class, salience DESC, updated_at DESC);
        CREATE TABLE IF NOT EXISTS person_profiles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            canonical_key TEXT NOT NULL UNIQUE,
            display_name TEXT NOT NULL,
            primary_email TEXT NOT NULL DEFAULT '',
            relationship_kind TEXT NOT NULL DEFAULT 'unknown',
            trust_level TEXT NOT NULL DEFAULT 'low',
            last_interaction_at TEXT,
            last_channel TEXT NOT NULL DEFAULT '',
            interaction_count INTEGER NOT NULL DEFAULT 0,
            conversation_memory_summary TEXT NOT NULL DEFAULT '',
            notebook_summary TEXT NOT NULL DEFAULT '',
            proactive_guard_note TEXT NOT NULL DEFAULT 'Proactive contact only after grosshirn validation and conflict-of-interest review.'
        );
        CREATE TABLE IF NOT EXISTS person_notes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_profile_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            note_kind TEXT NOT NULL,
            source_channel TEXT NOT NULL DEFAULT '',
            source_ref TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL,
            detail TEXT NOT NULL,
            important INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX IF NOT EXISTS idx_person_notes_profile_kind
            ON person_notes(person_profile_id, note_kind, updated_at DESC);
        CREATE TABLE IF NOT EXISTS proactive_contact_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            person_profile_id INTEGER,
            person_display_name TEXT NOT NULL,
            person_email TEXT NOT NULL DEFAULT '',
            source_task_id INTEGER,
            source_turn_id INTEGER,
            status TEXT NOT NULL DEFAULT 'pending_validation',
            channel TEXT NOT NULL,
            subject TEXT NOT NULL,
            draft_body TEXT NOT NULL,
            rationale TEXT NOT NULL,
            conflict_check TEXT NOT NULL DEFAULT '',
            requires_grosshirn_validation INTEGER NOT NULL DEFAULT 1,
            validation_task_id INTEGER,
            validated_at TEXT,
            validation_decision TEXT NOT NULL DEFAULT '',
            validation_note TEXT NOT NULL DEFAULT '',
            dispatch_task_id INTEGER,
            dispatched_at TEXT,
            dispatch_channel TEXT NOT NULL DEFAULT '',
            dispatch_note TEXT NOT NULL DEFAULT '',
            outbound_message_id TEXT NOT NULL DEFAULT '',
            source TEXT NOT NULL DEFAULT ''
        );
        CREATE INDEX IF NOT EXISTS idx_proactive_candidates_status
            ON proactive_contact_candidates(status, updated_at DESC);
        CREATE TABLE IF NOT EXISTS resources (
            category TEXT NOT NULL,
            name TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            status TEXT NOT NULL,
            detail TEXT NOT NULL,
            PRIMARY KEY(category, name)
        );
        CREATE TABLE IF NOT EXISTS skills (
            name TEXT PRIMARY KEY,
            path TEXT NOT NULL,
            status TEXT NOT NULL,
            notes TEXT NOT NULL,
            last_seen_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS homepage_revisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source_channel TEXT NOT NULL,
            title TEXT NOT NULL,
            headline TEXT NOT NULL,
            owner_branding_applied INTEGER NOT NULL DEFAULT 0,
            notes TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS bios_uploads (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            speaker TEXT NOT NULL,
            source_channel TEXT NOT NULL,
            note TEXT NOT NULL,
            file_name TEXT NOT NULL,
            public_path TEXT NOT NULL,
            mime_type TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS loop_interrupts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            source_channel TEXT NOT NULL,
            speaker TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending',
            claimed_at TEXT,
            processed_at TEXT,
            response TEXT,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_loop_interrupts_status_id
            ON loop_interrupts(status, id ASC);
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            parent_task_id INTEGER,
            worker_job_id INTEGER,
            source_interrupt_id INTEGER,
            source_channel TEXT NOT NULL,
            speaker TEXT NOT NULL,
            task_kind TEXT NOT NULL,
            title TEXT NOT NULL,
            detail TEXT NOT NULL,
            trust_level TEXT NOT NULL,
            priority_score INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'queued',
            run_count INTEGER NOT NULL DEFAULT 0,
            grosshirn_cooldown_until TEXT,
            last_checkpoint_summary TEXT,
            last_checkpoint_at TEXT,
            last_output TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tasks_status_priority_id
            ON tasks(status, priority_score DESC, id ASC);
        CREATE INDEX IF NOT EXISTS idx_tasks_kind_status_id
            ON tasks(task_kind, status, id DESC);
        CREATE TABLE IF NOT EXISTS task_checkpoints (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            task_id INTEGER NOT NULL,
            created_at TEXT NOT NULL,
            checkpoint_kind TEXT NOT NULL,
            summary TEXT NOT NULL,
            detail TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_task_checkpoints_task_id
            ON task_checkpoints(task_id, id DESC);
        CREATE TABLE IF NOT EXISTS focus_state (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            mode TEXT NOT NULL DEFAULT 'idle',
            active_task_id INTEGER,
            active_task_title TEXT NOT NULL DEFAULT '',
            queue_depth INTEGER NOT NULL DEFAULT 0,
            last_reprioritized_at TEXT,
            last_task_completed_at TEXT,
            note TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS bootstrap_task_seeds (
            seed_key TEXT PRIMARY KEY,
            task_id INTEGER NOT NULL,
            title TEXT NOT NULL,
            seeded_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS context_packages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            task_id INTEGER NOT NULL,
            task_title TEXT NOT NULL,
            context_mode TEXT NOT NULL,
            budget_hint INTEGER NOT NULL,
            rationale TEXT NOT NULL,
            package_json TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_context_packages_task_id
            ON context_packages(task_id, id DESC);
        CREATE TABLE IF NOT EXISTS context_embedding_chunks (
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
            ON context_embedding_chunks(source_kind, source_ref, embedding_model, updated_at DESC);
        CREATE TABLE IF NOT EXISTS mode_transitions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            from_mode TEXT NOT NULL,
            to_mode TEXT NOT NULL,
            active_task_id INTEGER,
            active_task_title TEXT NOT NULL,
            trigger TEXT NOT NULL,
            note TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS worker_jobs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            parent_task_id INTEGER NOT NULL,
            parent_task_title TEXT NOT NULL,
            worker_kind TEXT NOT NULL,
            contract_title TEXT NOT NULL,
            contract_detail TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued',
            request_note TEXT NOT NULL DEFAULT '',
            result_summary TEXT,
            result_detail TEXT,
            review_summary TEXT,
            review_detail TEXT,
            review_task_id INTEGER,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_worker_jobs_status_id
            ON worker_jobs(status, id ASC);
        CREATE TABLE IF NOT EXISTS agent_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            method TEXT NOT NULL,
            active_task_id INTEGER,
            active_task_title TEXT NOT NULL,
            body TEXT NOT NULL,
            payload_json TEXT NOT NULL DEFAULT '{}'
        );
        CREATE TABLE IF NOT EXISTS agent_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            task_id INTEGER NOT NULL,
            task_title TEXT NOT NULL,
            trigger TEXT NOT NULL,
            mode_at_start TEXT NOT NULL,
            mode_at_end TEXT,
            status TEXT NOT NULL DEFAULT 'in_progress',
            summary TEXT,
            output TEXT,
            completed_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_agent_turns_status_id
            ON agent_turns(status, id DESC);
        CREATE TABLE IF NOT EXISTS agent_threads (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            thread_key TEXT NOT NULL DEFAULT 'main',
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            lifecycle_status TEXT NOT NULL DEFAULT 'bootstrapping',
            current_mode TEXT NOT NULL DEFAULT 'observe',
            active_turn_id INTEGER,
            active_task_id INTEGER,
            queue_depth INTEGER NOT NULL DEFAULT 0,
            note TEXT NOT NULL DEFAULT ''
        );
        CREATE TABLE IF NOT EXISTS turn_signals (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            thread_key TEXT NOT NULL DEFAULT 'main',
            turn_id INTEGER,
            task_id INTEGER,
            signal_kind TEXT NOT NULL,
            source_channel TEXT NOT NULL,
            speaker TEXT NOT NULL,
            message TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'recorded',
            consumed_at TEXT,
            resolution_note TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_turn_signals_task_id
            ON turn_signals(task_id, id DESC);
        CREATE INDEX IF NOT EXISTS idx_turn_signals_status_id
            ON turn_signals(status, id ASC);
        CREATE TABLE IF NOT EXISTS loop_incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            incident_key TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'open',
            summary TEXT NOT NULL,
            detail TEXT NOT NULL,
            related_task_id INTEGER,
            related_turn_id INTEGER,
            self_preservation_task_id INTEGER,
            hard_reset_required INTEGER NOT NULL DEFAULT 0,
            hard_reset_report_path TEXT,
            resolved_at TEXT
        );
        "#,
    )?;
    add_column_if_missing(
        &conn,
        "tasks",
        "parent_task_id",
        "ALTER TABLE tasks ADD COLUMN parent_task_id INTEGER",
    )?;
    add_column_if_missing(
        &conn,
        "tasks",
        "worker_job_id",
        "ALTER TABLE tasks ADD COLUMN worker_job_id INTEGER",
    )?;
    add_column_if_missing(
        &conn,
        "tasks",
        "grosshirn_cooldown_until",
        "ALTER TABLE tasks ADD COLUMN grosshirn_cooldown_until TEXT",
    )?;
    add_column_if_missing(
        &conn,
        "proactive_contact_candidates",
        "dispatch_task_id",
        "ALTER TABLE proactive_contact_candidates ADD COLUMN dispatch_task_id INTEGER",
    )?;
    add_column_if_missing(
        &conn,
        "proactive_contact_candidates",
        "dispatched_at",
        "ALTER TABLE proactive_contact_candidates ADD COLUMN dispatched_at TEXT",
    )?;
    add_column_if_missing(
        &conn,
        "proactive_contact_candidates",
        "dispatch_channel",
        "ALTER TABLE proactive_contact_candidates ADD COLUMN dispatch_channel TEXT NOT NULL DEFAULT ''",
    )?;
    add_column_if_missing(
        &conn,
        "proactive_contact_candidates",
        "dispatch_note",
        "ALTER TABLE proactive_contact_candidates ADD COLUMN dispatch_note TEXT NOT NULL DEFAULT ''",
    )?;
    add_column_if_missing(
        &conn,
        "proactive_contact_candidates",
        "outbound_message_id",
        "ALTER TABLE proactive_contact_candidates ADD COLUMN outbound_message_id TEXT NOT NULL DEFAULT ''",
    )?;
    conn.execute("INSERT OR IGNORE INTO owner_trust(singleton) VALUES(1)", [])?;
    conn.execute(
        "INSERT OR IGNORE INTO brain_routing_state(singleton) VALUES(1)",
        [],
    )?;
    conn.execute("INSERT OR IGNORE INTO focus_state(singleton) VALUES(1)", [])?;
    conn.execute(
        "INSERT OR IGNORE INTO agent_threads(
            singleton, thread_key, created_at, updated_at, lifecycle_status, current_mode, queue_depth, note
         ) VALUES(1, 'main', ?1, ?1, 'bootstrapping', 'observe', 0, 'Main infinity thread bootstrapped.')",
        params![now_iso()],
    )?;
    conn.execute(
        "UPDATE tasks SET status = 'queued', updated_at = ?1 WHERE status = 'active'",
        params![now_iso()],
    )?;
    drop(conn);
    let _ = sanitize_unclean_runtime(paths)?;
    let _ = sync_owner_trust(paths)?;
    let _ = sync_skills(paths)?;
    Ok(())
}

pub fn seed_bootstrap_tasks(paths: &Paths) -> anyhow::Result<usize> {
    let pack = load_bootstrap_task_pack(paths);
    let mut conn = open_db(paths)?;
    let tx = conn.transaction()?;
    let mut seeded = 0_usize;

    for task in pack.tasks.iter().filter(|task| task.enabled) {
        let already_seeded = tx
            .query_row(
                "SELECT 1 FROM bootstrap_task_seeds WHERE seed_key = ?1",
                params![task.seed_key],
                |_| Ok(()),
            )
            .optional()?;
        if already_seeded.is_some() {
            continue;
        }

        let task_id = insert_bootstrap_task(&tx, task)?;
        tx.execute(
            "INSERT INTO bootstrap_task_seeds(seed_key, task_id, title, seeded_at)
             VALUES(?1, ?2, ?3, ?4)",
            params![task.seed_key, task_id, task.title, now_iso()],
        )?;
        seeded += 1;
    }

    tx.commit()?;

    if seeded > 0 {
        set_focus_state(
            paths,
            "observe",
            None,
            "",
            &format!("Bootstrapped {seeded} canonical startup tasks."),
        )?;
    }

    Ok(seeded)
}

pub fn sync_owner_trust(paths: &Paths) -> anyhow::Result<OwnerTrustSnapshot> {
    let bios = load_bios(paths);
    let root_auth = load_root_auth(paths);
    let conn = open_db(paths)?;
    let existing = load_owner_trust_row(&conn)?.unwrap_or_default();

    let owner_name = if bios.owner.name.trim().is_empty() {
        existing.owner_name.clone()
    } else {
        bios.owner.name.trim().to_string()
    };
    let committed_owner_name = existing.committed_owner_name.clone();
    let score = compute_owner_commitment_score(
        &owner_name,
        existing.owner_contact_established,
        existing.bios_primary_channel_confirmed,
        root_auth.configured,
    );

    conn.execute(
        "UPDATE owner_trust
         SET owner_name = ?1,
             committed_owner_name = ?2,
             superpassword_set = ?3,
             owner_commitment_score = ?4
         WHERE singleton = 1",
        params![
            owner_name,
            committed_owner_name,
            bool_to_i64(root_auth.configured),
            score,
        ],
    )?;

    load_owner_trust(paths)
}

pub fn load_owner_trust(paths: &Paths) -> anyhow::Result<OwnerTrustSnapshot> {
    let conn = open_db(paths)?;
    Ok(load_owner_trust_row(&conn)?.unwrap_or_default())
}

fn default_brain_routing_state() -> BrainRoutingState {
    BrainRoutingState {
        route_mode: "kleinhirn".to_string(),
        boosted_task_id: None,
        boosted_task_title: String::new(),
        boost_reason: String::new(),
        boost_started_at: None,
        boost_last_used_at: None,
        boost_expires_at: None,
        cooldown_until: None,
        last_deactivation_reason: String::new(),
    }
}

fn grosshirn_boost_ttl_secs() -> i64 {
    std::env::var("CTO_AGENT_GROSSHIRN_TASK_BOOST_TTL_SECS")
        .ok()
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value >= 30)
        .unwrap_or(300)
}

fn future_iso_after(secs: i64) -> String {
    (Utc::now() + ChronoDuration::seconds(secs)).to_rfc3339()
}

fn recent_past_iso(secs: i64) -> String {
    (Utc::now() - ChronoDuration::seconds(secs)).to_rfc3339()
}

fn blocked_internal_task_reuse_window_secs() -> i64 {
    1800
}

fn parse_rfc3339_utc(value: &str) -> Option<DateTime<Utc>> {
    DateTime::parse_from_rfc3339(value)
        .ok()
        .map(|parsed| parsed.with_timezone(&Utc))
}

fn load_task_grosshirn_cooldown_until_with_conn(
    conn: &Connection,
    task_id: i64,
) -> anyhow::Result<Option<String>> {
    conn.query_row(
        "SELECT grosshirn_cooldown_until FROM tasks WHERE id = ?1",
        params![task_id],
        |row| row.get::<_, Option<String>>(0),
    )
    .optional()
    .map(|value| value.flatten())
    .map_err(Into::into)
}

fn load_brain_routing_state_row(conn: &Connection) -> anyhow::Result<Option<BrainRoutingState>> {
    conn.query_row(
        "SELECT route_mode, boosted_task_id, boosted_task_title, boost_reason,
                boost_started_at, boost_last_used_at, boost_expires_at, cooldown_until,
                last_deactivation_reason
         FROM brain_routing_state
         WHERE singleton = 1",
        [],
        |row| {
            Ok(BrainRoutingState {
                route_mode: row.get(0)?,
                boosted_task_id: row.get(1)?,
                boosted_task_title: row.get(2)?,
                boost_reason: row.get(3)?,
                boost_started_at: row.get(4)?,
                boost_last_used_at: row.get(5)?,
                boost_expires_at: row.get(6)?,
                cooldown_until: row.get(7)?,
                last_deactivation_reason: row.get(8)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

pub fn load_brain_routing_state(paths: &Paths) -> anyhow::Result<BrainRoutingState> {
    let conn = open_db(paths)?;
    Ok(load_brain_routing_state_row(&conn)?.unwrap_or_else(default_brain_routing_state))
}

pub fn grosshirn_boost_available(paths: &Paths) -> bool {
    let trust = load_owner_trust(paths).unwrap_or_default();
    trust.brain_access_mode == "kleinhirn_plus_grosshirn"
        && crate::brain_runtime::grosshirn_runtime_configured(paths)
}

pub fn task_has_active_grosshirn_boost(paths: &Paths, task_id: i64) -> bool {
    let Ok(state) = load_brain_routing_state(paths) else {
        return false;
    };
    if state.route_mode != "grosshirn" || state.boosted_task_id != Some(task_id) {
        return false;
    }
    match state
        .boost_expires_at
        .as_deref()
        .and_then(parse_rfc3339_utc)
    {
        Some(deadline) => deadline > Utc::now(),
        None => true,
    }
}

fn task_is_in_grosshirn_cooldown_state(state: &BrainRoutingState, task_id: i64) -> bool {
    if state.route_mode != "kleinhirn" || state.boosted_task_id != Some(task_id) {
        return false;
    }
    match state.cooldown_until.as_deref().and_then(parse_rfc3339_utc) {
        Some(deadline) => deadline > Utc::now(),
        None => false,
    }
}

pub fn task_is_in_grosshirn_cooldown(paths: &Paths, task_id: i64) -> bool {
    let Ok(conn) = open_db(paths) else {
        return false;
    };
    if let Ok(Some(task_cooldown_until)) =
        load_task_grosshirn_cooldown_until_with_conn(&conn, task_id)
    {
        if parse_rfc3339_utc(task_cooldown_until.as_str())
            .map(|deadline| deadline > Utc::now())
            .unwrap_or(false)
        {
            return true;
        }
    }
    let Ok(state) = load_brain_routing_state_row(&conn) else {
        return false;
    };
    state
        .map(|state| task_is_in_grosshirn_cooldown_state(&state, task_id))
        .unwrap_or(false)
}

fn clear_stale_grosshirn_boost_for_selected_task(
    paths: &Paths,
    selected_task_id: i64,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    let state = load_brain_routing_state_row(&conn)?.unwrap_or_else(default_brain_routing_state);
    if state.route_mode != "grosshirn" {
        return Ok(());
    }
    let Some(boosted_task_id) = state.boosted_task_id else {
        return Ok(());
    };
    if boosted_task_id == selected_task_id {
        return Ok(());
    }

    let reason = format!(
        "Cleared stale grosshirn boost for task #{boosted_task_id} because task #{selected_task_id} became active instead."
    );
    conn.execute(
        "UPDATE brain_routing_state
         SET route_mode = 'kleinhirn',
             boosted_task_id = NULL,
             boosted_task_title = '',
             boost_reason = '',
             boost_started_at = NULL,
             boost_last_used_at = NULL,
             boost_expires_at = NULL,
             cooldown_until = NULL,
             last_deactivation_reason = ?1
         WHERE singleton = 1",
        params![reason],
    )?;
    drop(conn);
    let _ = record_resource_status(paths, "brain_routing", "active_route", "kleinhirn", &reason);
    Ok(())
}

pub fn arm_task_grosshirn_boost(
    paths: &Paths,
    task_id: i64,
    task_title: &str,
    reason: &str,
    priority_floor: i64,
) -> anyhow::Result<BrainRoutingState> {
    let conn = open_db(paths)?;
    let now = now_iso();
    let ttl_secs = grosshirn_boost_ttl_secs();
    let expires_at = future_iso_after(ttl_secs);
    conn.execute(
        "UPDATE brain_routing_state
         SET route_mode = 'grosshirn',
             boosted_task_id = ?1,
             boosted_task_title = ?2,
             boost_reason = ?3,
             boost_started_at = COALESCE(boost_started_at, ?4),
             boost_last_used_at = ?4,
             boost_expires_at = ?5,
             cooldown_until = NULL,
             last_deactivation_reason = ''
         WHERE singleton = 1",
        params![task_id, task_title, reason, now, expires_at],
    )?;
    conn.execute(
        "UPDATE tasks
         SET priority_score = CASE
             WHEN priority_score < ?2 THEN ?2
             ELSE priority_score
         END,
             grosshirn_cooldown_until = NULL,
             updated_at = ?3
         WHERE id = ?1",
        params![task_id, priority_floor, now],
    )?;
    drop(conn);
    let _ = record_resource_status(
        paths,
        "brain_routing",
        "active_route",
        "boosted",
        &format!("task #{task_id} :: {task_title} :: {reason}"),
    );
    load_brain_routing_state(paths)
}

pub fn refresh_task_grosshirn_boost(
    paths: &Paths,
    task_id: i64,
) -> anyhow::Result<BrainRoutingState> {
    let conn = open_db(paths)?;
    let state = load_brain_routing_state_row(&conn)?.unwrap_or_else(default_brain_routing_state);
    if state.route_mode != "grosshirn" || state.boosted_task_id != Some(task_id) {
        return Ok(state);
    }
    let now = now_iso();
    let expires_at = future_iso_after(grosshirn_boost_ttl_secs());
    conn.execute(
        "UPDATE brain_routing_state
         SET boost_last_used_at = ?1,
             boost_expires_at = ?2
         WHERE singleton = 1",
        params![now, expires_at],
    )?;
    drop(conn);
    load_brain_routing_state(paths)
}

pub fn release_task_grosshirn_boost(
    paths: &Paths,
    task_id: i64,
    reason: &str,
) -> anyhow::Result<BrainRoutingState> {
    let conn = open_db(paths)?;
    let state = load_brain_routing_state_row(&conn)?.unwrap_or_else(default_brain_routing_state);
    if state.boosted_task_id != Some(task_id) {
        return Ok(state);
    }
    let cooldown_until = future_iso_after(grosshirn_boost_ttl_secs());
    conn.execute(
        "UPDATE brain_routing_state
         SET route_mode = 'kleinhirn',
             boosted_task_id = NULL,
             boosted_task_title = '',
             boost_reason = '',
             boost_started_at = NULL,
             boost_last_used_at = NULL,
             boost_expires_at = NULL,
             cooldown_until = NULL,
             last_deactivation_reason = ?1
         WHERE singleton = 1",
        params![reason],
    )?;
    conn.execute(
        "UPDATE tasks
         SET grosshirn_cooldown_until = ?2,
             updated_at = ?3
         WHERE id = ?1",
        params![task_id, cooldown_until, now_iso()],
    )?;
    drop(conn);
    let _ = record_resource_status(paths, "brain_routing", "active_route", "kleinhirn", reason);
    load_brain_routing_state(paths)
}

pub fn expire_stale_grosshirn_boost(paths: &Paths) -> anyhow::Result<Option<String>> {
    let state = load_brain_routing_state(paths)?;
    if state.route_mode != "grosshirn" {
        return Ok(None);
    }
    let Some(task_id) = state.boosted_task_id else {
        return Ok(None);
    };
    let now = Utc::now();
    let expired = state
        .boost_expires_at
        .as_deref()
        .and_then(parse_rfc3339_utc)
        .map(|deadline| deadline <= now)
        .unwrap_or(false);
    let task_closed = load_task_by_id(paths, task_id)?
        .map(|task| !matches!(task.status.as_str(), "queued" | "active" | "await_review"))
        .unwrap_or(true);
    if !expired && !task_closed {
        return Ok(None);
    }
    let reason = if task_closed {
        format!("Grosshirn boost for task #{task_id} ended because the task is no longer open.")
    } else {
        format!(
            "Grosshirn boost for task #{task_id} fell back to kleinhirn after the cooldown expired."
        )
    };
    release_task_grosshirn_boost(paths, task_id, &reason)?;
    Ok(Some(reason))
}

pub fn set_brain_access_mode(paths: &Paths, mode: &str) -> anyhow::Result<OwnerTrustSnapshot> {
    let conn = open_db(paths)?;
    let normalized = match mode {
        "kleinhirn_plus_grosshirn" => "kleinhirn_plus_grosshirn",
        _ => "kleinhirn_only",
    };
    conn.execute(
        "UPDATE owner_trust SET brain_access_mode = ?1 WHERE singleton = 1",
        params![normalized],
    )?;
    load_owner_trust(paths)
}

pub fn record_brain_usage_event(
    paths: &Paths,
    task_id: Option<i64>,
    turn_id: Option<i64>,
    brain_tier: &str,
    source_label: &str,
    model_id: &str,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
    total_tokens: Option<i64>,
    estimated_cost_usd: Option<f64>,
    note: &str,
) -> anyhow::Result<i64> {
    let conn = open_db(paths)?;
    conn.execute(
        "INSERT INTO brain_usage_events(
            created_at, task_id, turn_id, brain_tier, source_label, model_id,
            input_tokens, output_tokens, total_tokens, estimated_cost_usd, note
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        params![
            now_iso(),
            task_id,
            turn_id,
            brain_tier.trim(),
            source_label.trim(),
            model_id.trim(),
            input_tokens.unwrap_or(0),
            output_tokens.unwrap_or(0),
            total_tokens.unwrap_or(0),
            estimated_cost_usd.unwrap_or(0.0),
            note.trim(),
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn load_external_brain_cost_summary(paths: &Paths) -> anyhow::Result<ExternalBrainCostSummary> {
    let conn = open_db(paths)?;
    let summary = conn.query_row(
        "SELECT
             COUNT(*),
             COALESCE(SUM(input_tokens), 0),
             COALESCE(SUM(output_tokens), 0),
             COALESCE(SUM(total_tokens), 0),
             COALESCE(SUM(estimated_cost_usd), 0.0),
             MAX(created_at)
         FROM brain_usage_events
         WHERE brain_tier = 'grosshirn'",
        [],
        |row| {
            Ok(ExternalBrainCostSummary {
                grosshirn_calls: row.get(0)?,
                total_input_tokens: row.get(1)?,
                total_output_tokens: row.get(2)?,
                total_tokens: row.get(3)?,
                estimated_cost_usd: row.get(4)?,
                last_external_call_at: row.get(5)?,
            })
        },
    )?;
    Ok(summary)
}

pub fn load_brain_usage_rollup(paths: &Paths) -> anyhow::Result<BrainUsageRollup> {
    let conn = open_db(paths)?;
    let (total_input_tokens, total_output_tokens, total_tokens) = conn.query_row(
        "SELECT
             COALESCE(SUM(input_tokens), 0),
             COALESCE(SUM(output_tokens), 0),
             COALESCE(SUM(total_tokens), 0)
         FROM brain_usage_events",
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
            ))
        },
    )?;

    let latest = conn
        .query_row(
            "SELECT model_id, brain_tier, input_tokens, output_tokens, total_tokens, created_at
             FROM brain_usage_events
             ORDER BY id DESC
             LIMIT 1",
            [],
            |row| {
                Ok(BrainUsageRollup {
                    total_input_tokens,
                    total_output_tokens,
                    total_tokens,
                    last_model_id: row.get(0)?,
                    last_brain_tier: row.get(1)?,
                    last_input_tokens: row.get(2)?,
                    last_output_tokens: row.get(3)?,
                    last_total_tokens: row.get(4)?,
                    last_recorded_at: row.get(5)?,
                })
            },
        )
        .optional()?;

    Ok(latest.unwrap_or(BrainUsageRollup {
        total_input_tokens,
        total_output_tokens,
        total_tokens,
        ..Default::default()
    }))
}

pub fn record_homepage_revision(
    paths: &Paths,
    source_channel: &str,
    policy: &HomepagePolicy,
    notes: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.execute(
        "INSERT INTO homepage_revisions(created_at, source_channel, title, headline, owner_branding_applied, notes)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            now_iso(),
            source_channel,
            policy.current_title,
            policy.current_headline,
            bool_to_i64(policy.owner_branding_applied),
            notes
        ],
    )?;
    Ok(())
}

pub fn record_bios_dialogue(
    paths: &Paths,
    speaker: &str,
    message: &str,
    used_grosshirn: bool,
) -> anyhow::Result<()> {
    let bios = load_bios(paths);
    let conn = open_db(paths)?;
    let created_at = now_iso();
    conn.execute(
        "INSERT INTO bios_dialogue(created_at, speaker, channel, message, used_grosshirn)
         VALUES(?1, ?2, 'bios', ?3, ?4)",
        params![
            created_at,
            speaker.trim(),
            message.trim(),
            bool_to_i64(used_grosshirn)
        ],
    )?;

    if is_owner_message(speaker, &bios.owner.name) {
        let existing = load_owner_trust_row(&conn)?.unwrap_or_default();
        let owner_name = if bios.owner.name.trim().is_empty() {
            existing.owner_name.clone()
        } else {
            bios.owner.name.trim().to_string()
        };
        let committed_owner_name = if !owner_name.is_empty() {
            owner_name.clone()
        } else {
            speaker.trim().to_string()
        };

        conn.execute(
            "UPDATE owner_trust
             SET owner_contact_established = 1,
                 bios_primary_channel_confirmed = 1,
                 owner_name = CASE WHEN owner_name = '' THEN ?1 ELSE owner_name END,
                 committed_owner_name = CASE WHEN committed_owner_name = '' THEN ?2 ELSE committed_owner_name END,
                 last_owner_dialogue_at = ?3
             WHERE singleton = 1",
            params![speaker.trim(), committed_owner_name, created_at],
        )?;

        conn.execute(
            "INSERT INTO memory_items(created_at, kind, summary, detail, source, important)
             VALUES(?1, 'owner_calibration', ?2, ?3, 'bios_chat', 1)",
            params![
                now_iso(),
                "Owner BIOS chat message received.",
                message.trim()
            ],
        )?;
        refresh_owner_memory_summary_with_conn(&conn)?;
    }

    let _ = sync_owner_trust(paths)?;
    Ok(())
}

pub fn record_memory(
    paths: &Paths,
    kind: &str,
    summary: &str,
    detail: &str,
    source: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.execute(
        "INSERT INTO memory_items(created_at, kind, summary, detail, source, important)
         VALUES(?1, ?2, ?3, ?4, ?5, 1)",
        params![now_iso(), kind, summary, detail, source],
    )?;
    Ok(())
}

pub fn record_progress_journal(
    paths: &Paths,
    summary: &str,
    detail: &str,
    source: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    record_progress_journal_with_conn(&conn, summary, detail, source)
}

pub fn record_terminal_feedback(
    paths: &Paths,
    speaker: &str,
    message: &str,
) -> anyhow::Result<Option<String>> {
    let bios = load_bios(paths);
    if !is_owner_message(speaker, &bios.owner.name) {
        return Ok(None);
    }

    let conn = open_db(paths)?;
    let created_at = now_iso();
    let existing = load_owner_trust_row(&conn)?.unwrap_or_default();
    let owner_name = if bios.owner.name.trim().is_empty() {
        existing.owner_name.clone()
    } else {
        bios.owner.name.trim().to_string()
    };

    conn.execute(
        "UPDATE owner_trust
         SET owner_contact_established = 1,
             owner_name = CASE WHEN owner_name = '' THEN ?1 ELSE owner_name END,
             last_owner_dialogue_at = ?2
         WHERE singleton = 1",
        params![
            if owner_name.is_empty() {
                speaker.trim()
            } else {
                owner_name.as_str()
            },
            created_at
        ],
    )?;

    let summary = summarize_for_memory(message);
    conn.execute(
        "INSERT INTO memory_items(created_at, kind, summary, detail, source, important)
         VALUES(?1, 'owner_calibration', ?2, ?3, 'terminal', 1)",
        params![now_iso(), summary, message.trim()],
    )?;
    refresh_owner_memory_summary_with_conn(&conn)?;
    let trust = sync_owner_trust(paths)?;

    Ok(Some(format!(
        "Owner terminal contact registered. Commitment is now {}/100.",
        trust.owner_commitment_score
    )))
}

pub fn record_owner_chat_contact(
    paths: &Paths,
    source_channel: &str,
    speaker: &str,
) -> anyhow::Result<Option<String>> {
    let bios = load_bios(paths);
    if !is_owner_message(speaker, &bios.owner.name) {
        return Ok(None);
    }

    let conn = open_db(paths)?;
    let created_at = now_iso();
    let existing = load_owner_trust_row(&conn)?.unwrap_or_default();
    let owner_name = if bios.owner.name.trim().is_empty() {
        existing.owner_name.clone()
    } else {
        bios.owner.name.trim().to_string()
    };

    conn.execute(
        "UPDATE owner_trust
         SET owner_contact_established = 1,
             owner_name = CASE WHEN owner_name = '' THEN ?1 ELSE owner_name END,
             last_owner_dialogue_at = ?2
         WHERE singleton = 1",
        params![
            if owner_name.is_empty() {
                speaker.trim()
            } else {
                owner_name.as_str()
            },
            created_at
        ],
    )?;

    let source_label = match source_channel {
        "attach_terminal" => "TUI chat",
        "bios" => "BIOS chat",
        _ => "chat",
    };
    conn.execute(
        "INSERT INTO memory_items(created_at, kind, summary, detail, source, important)
         VALUES(?1, 'owner_calibration', ?2, ?3, ?4, 1)",
        params![
            now_iso(),
            format!("Owner contact established via {source_label}."),
            format!("Owner contact established via {source_label}."),
            source_channel,
        ],
    )?;
    refresh_owner_memory_summary_with_conn(&conn)?;
    let trust = sync_owner_trust(paths)?;

    Ok(Some(format!(
        "Owner {} registered. Commitment is now {}/100.",
        source_label, trust.owner_commitment_score
    )))
}

pub fn list_bios_dialogue(paths: &Paths, limit: usize) -> anyhow::Result<Vec<BiosDialogueEntry>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT created_at, speaker, message, used_grosshirn
         FROM bios_dialogue
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        Ok(BiosDialogueEntry {
            created_at: row.get(0)?,
            speaker: row.get(1)?,
            message: row.get(2)?,
            used_grosshirn: row.get::<_, i64>(3)? != 0,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_memory_items(paths: &Paths, limit: usize) -> anyhow::Result<Vec<MemoryItemRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT created_at, kind, summary, detail, source, important
         FROM memory_items
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        Ok(MemoryItemRecord {
            created_at: row.get(0)?,
            kind: row.get(1)?,
            summary: row.get(2)?,
            detail: row.get(3)?,
            source: row.get(4)?,
            important: row.get::<_, i64>(5)? != 0,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_active_learning_entries(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<LearningEntryRecord>> {
    let conn = open_db(paths)?;
    list_active_learning_entries_with_conn(&conn, limit)
}

pub fn list_person_profiles(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<PersonProfileRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, canonical_key, display_name, primary_email,
                relationship_kind, trust_level, last_interaction_at, last_channel,
                interaction_count, conversation_memory_summary, notebook_summary,
                proactive_guard_note
         FROM person_profiles
         ORDER BY CASE relationship_kind
                    WHEN 'owner' THEN 0
                    WHEN 'reports_to' THEN 1
                    WHEN 'ceo' THEN 2
                    WHEN 'board' THEN 3
                    WHEN 'peer_cxo' THEN 4
                    WHEN 'subordinate_person' THEN 5
                    WHEN 'vendor' THEN 6
                    ELSE 7
                  END,
                  interaction_count DESC,
                  updated_at DESC,
                  id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_person_profile_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_recent_person_notes(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<PersonNoteRecord>> {
    let conn = open_db(paths)?;
    list_recent_person_notes_with_conn(&conn, limit)
}

pub fn list_person_notes_for_person(
    paths: &Paths,
    person_profile_id: i64,
    limit: usize,
) -> anyhow::Result<Vec<PersonNoteRecord>> {
    let conn = open_db(paths)?;
    list_person_notes_for_person_with_conn(&conn, person_profile_id, limit)
}

pub fn list_proactive_contact_candidates(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<ProactiveContactCandidateRecord>> {
    let conn = open_db(paths)?;
    list_proactive_contact_candidates_with_conn(&conn, limit, false)
}

pub fn list_pending_proactive_contact_candidates(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<ProactiveContactCandidateRecord>> {
    let conn = open_db(paths)?;
    list_proactive_contact_candidates_with_conn(&conn, limit, true)
}

pub fn load_proactive_contact_candidate_by_dispatch_task(
    paths: &Paths,
    dispatch_task_id: i64,
) -> anyhow::Result<Option<ProactiveContactCandidateRecord>> {
    let conn = open_db(paths)?;
    load_proactive_contact_candidate_by_dispatch_task_with_conn(&conn, dispatch_task_id)
}

pub fn list_recent_mail_previews_for_person(
    paths: &Paths,
    email: &str,
    limit: usize,
) -> anyhow::Result<Vec<MailPreviewRecord>> {
    let normalized_email = normalize_email_address(email);
    if normalized_email.is_empty() || limit == 0 {
        return Ok(Vec::new());
    }
    let conn = open_db(paths)?;
    let like_pattern = format!("%{}%", normalized_email);
    if table_exists(&conn, "communication_messages")? {
        let mut stmt = conn.prepare(
            "SELECT external_created_at, direction, subject, preview
             FROM communication_messages
             WHERE channel = 'email'
               AND (
                    lower(sender_address) = ?1
                    OR lower(recipient_addresses_json) LIKE ?2
                    OR lower(cc_addresses_json) LIKE ?2
               )
             ORDER BY external_created_at DESC, observed_at DESC
             LIMIT ?3",
        )?;
        let rows = stmt.query_map(
            params![normalized_email, like_pattern, limit as i64],
            |row| {
                Ok(MailPreviewRecord {
                    received_at_iso: row.get(0)?,
                    direction: row.get(1)?,
                    subject: row.get(2)?,
                    preview: row.get(3)?,
                })
            },
        )?;
        return Ok(rows.filter_map(Result::ok).collect());
    }
    if !table_exists(&conn, "mail_messages")? {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT received_at_iso, direction, subject, preview
         FROM mail_messages
         WHERE lower(from_email) = ?1
            OR lower(to_emails_json) LIKE ?2
            OR lower(cc_emails_json) LIKE ?2
         ORDER BY received_at_ts DESC
         LIMIT ?3",
    )?;
    let rows = stmt.query_map(
        params![normalized_email, like_pattern, limit as i64],
        |row| {
            Ok(MailPreviewRecord {
                received_at_iso: row.get(0)?,
                direction: row.get(1)?,
                subject: row.get(2)?,
                preview: row.get(3)?,
            })
        },
    )?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn communication_message_sent(paths: &Paths, message_id: &str) -> anyhow::Result<bool> {
    let normalized = message_id.trim();
    if normalized.is_empty() {
        return Ok(false);
    }
    let conn = open_db(paths)?;
    if !table_exists(&conn, "communication_messages")? {
        return Ok(false);
    }
    let count: i64 = conn.query_row(
        "SELECT COUNT(*)
         FROM communication_messages
         WHERE channel = 'email'
           AND direction = 'outbound'
           AND status = 'sent'
           AND (
                remote_id = ?1
                OR json_extract(metadata_json, '$.messageId') = ?1
           )",
        params![normalized],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

pub fn communication_email_sync_needs_baseline(paths: &Paths) -> anyhow::Result<bool> {
    let conn = open_db(paths)?;
    if !table_exists(&conn, "communication_sync_runs")? {
        return Ok(true);
    }
    let successful_runs: i64 = conn.query_row(
        "SELECT COUNT(*)
         FROM communication_sync_runs
         WHERE channel = 'email'
           AND ok = 1",
        [],
        |row| row.get(0),
    )?;
    Ok(successful_runs == 0)
}

pub fn record_person_interaction(
    paths: &Paths,
    source_channel: &str,
    speaker: &str,
    message: &str,
    source_ref: &str,
) -> anyhow::Result<Option<PersonProfileRecord>> {
    let conn = open_db(paths)?;
    record_person_interaction_with_conn(&conn, paths, source_channel, speaker, message, source_ref)
}

pub fn store_proactive_contact_candidate(
    paths: &Paths,
    task: &TaskRecord,
    turn_id: i64,
    draft: &ProactiveContactDraft,
    requires_grosshirn_validation: bool,
    source: &str,
) -> anyhow::Result<ProactiveContactCandidateRecord> {
    let conn = open_db(paths)?;
    let timestamp = now_iso();
    let person_probe = if !draft.person_email.trim().is_empty() {
        draft.person_email.trim()
    } else {
        draft.person_name.trim()
    };
    let profile = resolve_person_identity(paths, person_probe)
        .map(|identity| {
            upsert_person_profile_with_conn(&conn, &identity, "proactive_candidate", None)
        })
        .transpose()?
        .flatten();
    let person_profile_id = profile.as_ref().map(|record| record.id);
    let person_display_name = first_non_empty(&[
        draft.person_name.trim(),
        profile
            .as_ref()
            .map(|record| record.display_name.as_str())
            .unwrap_or(""),
        draft.person_email.trim(),
    ])
    .to_string();
    let person_email = first_non_empty(&[
        draft.person_email.trim(),
        profile
            .as_ref()
            .map(|record| record.primary_email.as_str())
            .unwrap_or(""),
    ])
    .to_string();
    let channel = normalize_proactive_contact_channel(&draft.channel, &person_email);
    let subject = summarize_for_memory(draft.subject.trim());
    let draft_body = draft.body.trim().to_string();
    let rationale = draft.rationale.trim().to_string();
    let conflict_check = if draft.conflict_check.trim().is_empty() {
        "No explicit conflict-of-interest review recorded yet.".to_string()
    } else {
        draft.conflict_check.trim().to_string()
    };
    let existing_id = conn
        .query_row(
            "SELECT id
             FROM proactive_contact_candidates
             WHERE status IN ('pending_validation', 'needs_revision')
               AND lower(person_display_name) = ?1
               AND lower(person_email) = ?2
               AND lower(subject) = ?3
             ORDER BY updated_at DESC, id DESC
             LIMIT 1",
            params![
                person_display_name.to_lowercase(),
                person_email.to_lowercase(),
                subject.to_lowercase(),
            ],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    let candidate_id = if let Some(candidate_id) = existing_id {
        conn.execute(
            "UPDATE proactive_contact_candidates
             SET updated_at = ?2,
                 person_profile_id = COALESCE(person_profile_id, ?3),
                 source_task_id = ?4,
                 source_turn_id = ?5,
                 status = 'pending_validation',
                 channel = ?6,
                 subject = ?7,
                 draft_body = ?8,
                 rationale = ?9,
                 conflict_check = ?10,
                 requires_grosshirn_validation = ?11,
                 validation_decision = '',
                 validation_note = '',
                 validated_at = NULL,
                 dispatch_task_id = NULL,
                 dispatched_at = NULL,
                 dispatch_channel = '',
                 dispatch_note = '',
                 outbound_message_id = '',
                 source = ?12
             WHERE id = ?1",
            params![
                candidate_id,
                timestamp,
                person_profile_id,
                task.id,
                turn_id,
                channel,
                subject,
                draft_body,
                rationale,
                conflict_check,
                bool_to_i64(requires_grosshirn_validation),
                source,
            ],
        )?;
        candidate_id
    } else {
        conn.execute(
            "INSERT INTO proactive_contact_candidates(
                created_at, updated_at, person_profile_id, person_display_name, person_email,
                source_task_id, source_turn_id, status, channel, subject, draft_body, rationale,
                conflict_check, requires_grosshirn_validation, validation_task_id, validated_at,
                validation_decision, validation_note, dispatch_task_id, dispatched_at,
                dispatch_channel, dispatch_note, outbound_message_id, source
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending_validation', ?8, ?9, ?10, ?11, ?12, ?13, NULL, NULL, '', '', NULL, NULL, '', '', '', ?14)",
            params![
                timestamp,
                timestamp,
                person_profile_id,
                person_display_name,
                person_email,
                task.id,
                turn_id,
                channel,
                subject,
                draft_body,
                rationale,
                conflict_check,
                bool_to_i64(requires_grosshirn_validation),
                source,
            ],
        )?;
        conn.last_insert_rowid()
    };
    load_proactive_contact_candidate_by_id_with_conn(&conn, candidate_id)?.ok_or_else(|| {
        anyhow::anyhow!("failed to reload proactive contact candidate {candidate_id}")
    })
}

pub fn attach_validation_task_to_candidate(
    paths: &Paths,
    candidate_id: i64,
    validation_task_id: i64,
) -> anyhow::Result<Option<ProactiveContactCandidateRecord>> {
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE proactive_contact_candidates
         SET updated_at = ?2,
             validation_task_id = CASE
                 WHEN validation_task_id IS NULL THEN ?3
                 ELSE validation_task_id
             END
         WHERE id = ?1",
        params![candidate_id, now_iso(), validation_task_id],
    )?;
    load_proactive_contact_candidate_by_id_with_conn(&conn, candidate_id)
}

pub fn attach_dispatch_task_to_candidate(
    paths: &Paths,
    candidate_id: i64,
    dispatch_task_id: i64,
) -> anyhow::Result<Option<ProactiveContactCandidateRecord>> {
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE proactive_contact_candidates
         SET updated_at = ?2,
             dispatch_task_id = CASE
                 WHEN dispatch_task_id IS NULL THEN ?3
                 ELSE dispatch_task_id
             END
         WHERE id = ?1",
        params![candidate_id, now_iso(), dispatch_task_id],
    )?;
    load_proactive_contact_candidate_by_id_with_conn(&conn, candidate_id)
}

pub fn apply_proactive_contact_validation(
    paths: &Paths,
    validation_task_id: i64,
    validation: &ProactiveContactValidationDraft,
    source: &str,
) -> anyhow::Result<Option<ProactiveContactCandidateRecord>> {
    let conn = open_db(paths)?;
    let Some(existing) =
        load_proactive_contact_candidate_by_validation_task_with_conn(&conn, validation_task_id)?
    else {
        return Ok(None);
    };
    let decision = normalize_proactive_validation_decision(&validation.decision);
    let status = match decision.as_str() {
        "approve" => "approved",
        "reject" => "rejected",
        _ => "needs_revision",
    };
    let subject = first_non_empty(&[validation.revised_subject.trim(), existing.subject.as_str()])
        .to_string();
    let draft_body =
        first_non_empty(&[validation.revised_body.trim(), existing.draft_body.as_str()])
            .to_string();
    let validation_note = if validation.note.trim().is_empty() {
        "No explicit validation note was recorded.".to_string()
    } else {
        validation.note.trim().to_string()
    };
    conn.execute(
        "UPDATE proactive_contact_candidates
         SET updated_at = ?2,
             status = ?3,
             subject = ?4,
             draft_body = ?5,
             validated_at = ?6,
             validation_decision = ?7,
             validation_note = ?8,
             dispatch_task_id = NULL,
             dispatched_at = NULL,
             dispatch_channel = '',
             dispatch_note = '',
             outbound_message_id = '',
             source = ?9
         WHERE id = ?1",
        params![
            existing.id,
            now_iso(),
            status,
            subject,
            draft_body,
            now_iso(),
            decision,
            validation_note,
            source,
        ],
    )?;
    let updated = load_proactive_contact_candidate_by_id_with_conn(&conn, existing.id)?;
    if let Some(candidate) = updated.as_ref() {
        if let Some(person_profile_id) = candidate.person_profile_id {
            let summary = match candidate.status.as_str() {
                "approved" => format!("Approved proactive proposal: {}", candidate.subject),
                "rejected" => format!("Rejected proactive proposal: {}", candidate.subject),
                _ => format!("Revised proactive proposal: {}", candidate.subject),
            };
            let detail = format!(
                "Channel: {channel}\nDecision: {decision}\nValidation Note: {note}\nRationale: {rationale}\nConflict Check: {conflict_check}\n\nDraft:\n{body}",
                channel = candidate.channel,
                decision = candidate.validation_decision,
                note = candidate.validation_note,
                rationale = candidate.rationale,
                conflict_check = candidate.conflict_check,
                body = candidate.draft_body,
            );
            record_person_note_with_conn(
                &conn,
                person_profile_id,
                "notebook",
                "proactive_review",
                &format!("proactive_candidate:{}", candidate.id),
                &summary,
                &detail,
                candidate.status == "approved",
            )?;
            refresh_person_profile_summaries_with_conn(&conn, person_profile_id)?;
            refresh_people_memory_summary_with_conn(&conn)?;
        }
    }
    Ok(updated)
}

pub fn record_proactive_contact_dispatch_result(
    paths: &Paths,
    dispatch_task_id: i64,
    status: &str,
    dispatch_channel: &str,
    dispatch_note: &str,
    outbound_message_id: &str,
    source: &str,
) -> anyhow::Result<Option<ProactiveContactCandidateRecord>> {
    let conn = open_db(paths)?;
    let Some(existing) =
        load_proactive_contact_candidate_by_dispatch_task_with_conn(&conn, dispatch_task_id)?
    else {
        return Ok(None);
    };
    let normalized_status = normalize_proactive_dispatch_status(status);
    let normalized_channel = normalize_proactive_contact_channel(
        if dispatch_channel.trim().is_empty() {
            existing.channel.as_str()
        } else {
            dispatch_channel
        },
        &existing.person_email,
    );
    let note = if dispatch_note.trim().is_empty() {
        "No explicit dispatch note was recorded.".to_string()
    } else {
        dispatch_note.trim().to_string()
    };
    let message_id = outbound_message_id.trim().to_string();
    let dispatched_at = if normalized_status == "sent" {
        Some(now_iso())
    } else {
        existing.dispatched_at.clone()
    };
    conn.execute(
        "UPDATE proactive_contact_candidates
         SET updated_at = ?2,
             status = ?3,
             dispatched_at = ?4,
             dispatch_channel = ?5,
             dispatch_note = ?6,
             outbound_message_id = ?7,
             source = ?8
         WHERE id = ?1",
        params![
            existing.id,
            now_iso(),
            normalized_status,
            dispatched_at,
            normalized_channel,
            note,
            message_id,
            source,
        ],
    )?;
    let updated = load_proactive_contact_candidate_by_id_with_conn(&conn, existing.id)?;
    if let Some(candidate) = updated.as_ref() {
        if let Some(person_profile_id) = candidate.person_profile_id {
            let source_ref = format!("proactive_dispatch:{}", candidate.id);
            let notebook_summary = match candidate.status.as_str() {
                "sent" => format!("Proactive message sent: {}", candidate.subject),
                "dispatch_blocked" => {
                    format!("Proactive dispatch blocked: {}", candidate.subject)
                }
                _ => format!("Proactive dispatch failed: {}", candidate.subject),
            };
            let notebook_detail = format!(
                "Intent Channel: {intent_channel}\nDispatch Channel: {dispatch_channel}\nDispatch Status: {status}\nDispatch Note: {note}\nMessage-ID: {message_id}\nValidation Note: {validation_note}\nRationale: {rationale}\nConflict Check: {conflict_check}\n\nDraft:\n{body}",
                intent_channel = candidate.channel,
                dispatch_channel = if candidate.dispatch_channel.trim().is_empty() {
                    "unknown"
                } else {
                    candidate.dispatch_channel.as_str()
                },
                status = candidate.status,
                note = candidate.dispatch_note,
                message_id = if candidate.outbound_message_id.trim().is_empty() {
                    "none"
                } else {
                    candidate.outbound_message_id.as_str()
                },
                validation_note = candidate.validation_note,
                rationale = candidate.rationale,
                conflict_check = candidate.conflict_check,
                body = candidate.draft_body,
            );
            record_person_note_with_conn(
                &conn,
                person_profile_id,
                "notebook",
                "proactive_dispatch",
                &source_ref,
                &notebook_summary,
                &notebook_detail,
                candidate.status == "sent",
            )?;
            if candidate.status == "sent" {
                let conversation_summary = format!(
                    "Contacted proactively via {}: {}",
                    candidate.dispatch_channel, candidate.subject
                );
                let conversation_detail = format!(
                    "Proactive message was sent autonomously.\n\nDispatch Channel: {dispatch_channel}\nMessage-ID: {message_id}\nSubject: {subject}\n\nBody:\n{body}",
                    dispatch_channel = candidate.dispatch_channel,
                    message_id = if candidate.outbound_message_id.trim().is_empty() {
                        "none"
                    } else {
                        candidate.outbound_message_id.as_str()
                    },
                    subject = candidate.subject,
                    body = candidate.draft_body,
                );
                record_person_note_with_conn(
                    &conn,
                    person_profile_id,
                    "conversation",
                    candidate.dispatch_channel.as_str(),
                    &source_ref,
                    &conversation_summary,
                    &conversation_detail,
                    true,
                )?;
                conn.execute(
                    "UPDATE person_profiles
                     SET updated_at = ?2,
                         last_interaction_at = ?2,
                         last_channel = ?3,
                         interaction_count = interaction_count + 1
                     WHERE id = ?1",
                    params![
                        person_profile_id,
                        now_iso(),
                        candidate.dispatch_channel.as_str()
                    ],
                )?;
            }
            refresh_person_profile_summaries_with_conn(&conn, person_profile_id)?;
            refresh_people_memory_summary_with_conn(&conn)?;
        }
    }
    Ok(updated)
}

pub fn store_learning_entries(
    paths: &Paths,
    task: &TaskRecord,
    turn_id: i64,
    entries: &[LearningEntryDraft],
    status: &str,
    source: &str,
) -> anyhow::Result<Vec<LearningEntryRecord>> {
    if entries.is_empty() {
        return Ok(Vec::new());
    }
    let conn = open_db(paths)?;
    let mut stored = Vec::new();
    for entry in entries {
        let summary = entry.summary.trim();
        if summary.is_empty() {
            continue;
        }
        let learning_class = normalize_learning_class(&entry.learning_class);
        let detail = if entry.detail.trim().is_empty() {
            summary.to_string()
        } else {
            entry.detail.trim().to_string()
        };
        let evidence = if entry.evidence.trim().is_empty() {
            "No explicit evidence recorded.".to_string()
        } else {
            entry.evidence.trim().to_string()
        };
        let applicability = if entry.applicability.trim().is_empty() {
            "No concrete usage context recorded.".to_string()
        } else {
            entry.applicability.trim().to_string()
        };
        let confidence = entry.confidence.clamp(0.05, 1.0);
        let salience = entry.salience.clamp(1, 100);
        let timestamp = now_iso();
        let existing_id = conn
            .query_row(
                "SELECT id
                 FROM learning_entries
                 WHERE learning_class = ?1
                   AND summary = ?2
                   AND status IN ('candidate', 'active')
                 ORDER BY CASE status WHEN 'active' THEN 0 ELSE 1 END, updated_at DESC, id DESC
                 LIMIT 1",
                params![learning_class, summary],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if let Some(entry_id) = existing_id {
            conn.execute(
                "UPDATE learning_entries
                 SET updated_at = ?2,
                     source_task_id = COALESCE(source_task_id, ?3),
                     source_turn_id = COALESCE(source_turn_id, ?4),
                     source_task_kind = CASE WHEN source_task_kind = '' THEN ?5 ELSE source_task_kind END,
                     source_channel = CASE WHEN source_channel = '' THEN ?6 ELSE source_channel END,
                     status = CASE
                         WHEN status = 'active' THEN 'active'
                         ELSE ?7
                     END,
                     detail = ?8,
                     evidence = ?9,
                     applicability = ?10,
                     confidence = ?11,
                     salience = CASE WHEN salience > ?12 THEN salience ELSE ?12 END,
                     source = CASE WHEN source = '' THEN ?13 ELSE source END
                 WHERE id = ?1",
                params![
                    entry_id,
                    timestamp,
                    task.id,
                    turn_id,
                    task.task_kind,
                    task.source_channel,
                    status,
                    detail,
                    evidence,
                    applicability,
                    confidence,
                    salience,
                    source,
                ],
            )?;
            if let Some(updated) = load_learning_entry_by_id_with_conn(&conn, entry_id)? {
                stored.push(updated);
            }
            continue;
        }

        conn.execute(
            "INSERT INTO learning_entries(
                created_at, updated_at, source_task_id, source_turn_id, source_task_kind, source_channel,
                learning_class, status, summary, detail, evidence, applicability, confidence, salience,
                recall_count, last_recalled_at, source
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, 0, NULL, ?15)",
            params![
                timestamp,
                timestamp,
                task.id,
                turn_id,
                task.task_kind,
                task.source_channel,
                learning_class,
                status,
                summary,
                detail,
                evidence,
                applicability,
                confidence,
                salience,
                source,
            ],
        )?;
        let entry_id = conn.last_insert_rowid();
        if let Some(inserted) = load_learning_entry_by_id_with_conn(&conn, entry_id)? {
            stored.push(inserted);
        }
    }

    if stored.iter().any(|entry| entry.status == "active") {
        refresh_learning_memory_summaries_with_conn(&conn)?;
        let _ = store_person_learning_refs_with_conn(paths, &conn, task, &stored);
    }
    Ok(stored)
}

pub fn promote_learning_candidates_for_task(
    paths: &Paths,
    task_id: i64,
    source: &str,
) -> anyhow::Result<usize> {
    let conn = open_db(paths)?;
    let updated_at = now_iso();
    let changed = conn.execute(
        "UPDATE learning_entries
         SET status = 'active',
             updated_at = ?2,
             source = CASE WHEN source = '' THEN ?3 ELSE source END
         WHERE source_task_id = ?1
           AND status = 'candidate'",
        params![task_id, updated_at, source],
    )?;
    if changed > 0 {
        refresh_learning_memory_summaries_with_conn(&conn)?;
        if let Some(task) = load_task_by_id(paths, task_id)? {
            let mut stmt = conn.prepare(
                "SELECT id, created_at, updated_at, source_task_id, source_turn_id, source_task_kind,
                        source_channel, learning_class, status, summary, detail, evidence, applicability,
                        confidence, salience, recall_count, last_recalled_at, source
                 FROM learning_entries
                 WHERE source_task_id = ?1
                   AND status = 'active'
                 ORDER BY salience DESC, confidence DESC, updated_at DESC, id DESC",
            )?;
            let rows = stmt.query_map(params![task_id], map_learning_entry_row)?;
            let active = rows.filter_map(Result::ok).collect::<Vec<_>>();
            let _ = store_person_learning_refs_with_conn(paths, &conn, &task, &active);
        }
    }
    Ok(changed)
}

pub fn reject_learning_candidates_for_task(
    paths: &Paths,
    task_id: i64,
    source: &str,
) -> anyhow::Result<usize> {
    let conn = open_db(paths)?;
    let changed = conn.execute(
        "UPDATE learning_entries
         SET status = 'rejected',
             updated_at = ?2,
             source = CASE WHEN source = '' THEN ?3 ELSE source END
         WHERE source_task_id = ?1
           AND status = 'candidate'",
        params![task_id, now_iso(), source],
    )?;
    Ok(changed)
}

pub fn mark_learning_entries_recalled(paths: &Paths, entry_ids: &[i64]) -> anyhow::Result<()> {
    if entry_ids.is_empty() {
        return Ok(());
    }
    let conn = open_db(paths)?;
    let timestamp = now_iso();
    for entry_id in entry_ids {
        conn.execute(
            "UPDATE learning_entries
             SET recall_count = recall_count + 1,
                 last_recalled_at = ?2
             WHERE id = ?1
               AND status = 'active'",
            params![entry_id, timestamp],
        )?;
    }
    Ok(())
}

pub fn load_memory_summary(paths: &Paths, scope: &str) -> anyhow::Result<Option<String>> {
    let conn = open_db(paths)?;
    let summary = conn
        .query_row(
            "SELECT summary FROM memory_summaries WHERE scope = ?1",
            params![scope],
            |row| row.get(0),
        )
        .optional()?;
    Ok(summary)
}

pub fn sync_resources_from_census(
    paths: &Paths,
    census: &SystemCensus,
) -> anyhow::Result<Vec<ResourceRecord>> {
    let conn = open_db(paths)?;
    let observed_at = census.captured_at.clone().unwrap_or_else(now_iso);

    upsert_resource(
        &conn,
        "host",
        "platform",
        &observed_at,
        "observed",
        &census
            .platform
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
    )?;
    upsert_resource(
        &conn,
        "host",
        "cpu_threads",
        &observed_at,
        "observed",
        &census
            .cpu_threads
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
    )?;
    upsert_resource(
        &conn,
        "host",
        "total_memory_gb",
        &observed_at,
        "observed",
        &census
            .total_memory_gb
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
    )?;
    upsert_resource(
        &conn,
        "host",
        "gpu_count",
        &observed_at,
        "observed",
        &census
            .gpu_count
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
    )?;
    upsert_resource(
        &conn,
        "host",
        "total_gpu_memory_gb",
        &observed_at,
        "observed",
        &census
            .total_gpu_memory_gb
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
    )?;
    upsert_resource(
        &conn,
        "host",
        "max_single_gpu_memory_gb",
        &observed_at,
        "observed",
        &census
            .max_single_gpu_memory_gb
            .map(|v| v.to_string())
            .unwrap_or_else(|| "unknown".to_string()),
    )?;
    upsert_resource(
        &conn,
        "host",
        "hostname",
        &observed_at,
        "observed",
        &census
            .hostname
            .clone()
            .unwrap_or_else(|| "unknown".to_string()),
    )?;
    if let Some(gpus) = &census.gpus {
        let inventory = gpus
            .iter()
            .map(|gpu| format!("#{} {} ({} MiB)", gpu.index, gpu.name, gpu.memory_total_mb))
            .collect::<Vec<_>>()
            .join(", ");
        upsert_resource(
            &conn,
            "host",
            "gpu_inventory",
            &observed_at,
            "observed",
            if inventory.is_empty() {
                "none"
            } else {
                &inventory
            },
        )?;
    }
    if let Some(candidates) = &census.model_tune_candidates {
        let summary = candidates
            .iter()
            .map(|candidate| format!("{}:{}", candidate.official_label, candidate.status))
            .collect::<Vec<_>>()
            .join(", ");
        upsert_resource(
            &conn,
            "model",
            "mistralrs_tune_candidates",
            &observed_at,
            "observed",
            if summary.is_empty() { "none" } else { &summary },
        )?;
    }

    list_resources(paths, 12)
}

pub fn sync_model_resources(
    paths: &Paths,
    policy: &ModelPolicy,
    census: &SystemCensus,
) -> anyhow::Result<Vec<ResourceRecord>> {
    let conn = open_db(paths)?;
    let observed_at = now_iso();
    upsert_resource(
        &conn,
        "model",
        "kleinhirn_basis",
        &observed_at,
        "policy",
        &format!(
            "{} ({})",
            policy.kleinhirn.official_label, policy.kleinhirn.model_id
        ),
    )?;
    upsert_resource(
        &conn,
        "model",
        "kleinhirn_selected",
        &observed_at,
        "policy",
        &describe_kleinhirn_selection(policy, census),
    )?;

    let brain_access_mode = load_owner_trust(paths)
        .map(|trust| trust.brain_access_mode)
        .unwrap_or_else(|_| "kleinhirn_only".to_string());
    upsert_resource(
        &conn,
        "model",
        "brain_access_mode",
        &observed_at,
        "policy",
        &brain_access_mode,
    )?;
    let grosshirn_candidates = if policy.grosshirn_candidates.is_empty() {
        "none".to_string()
    } else {
        policy
            .grosshirn_candidates
            .iter()
            .map(|candidate| format!("{} ({})", candidate.official_label, candidate.model_id))
            .collect::<Vec<_>>()
            .join(", ")
    };
    upsert_resource(
        &conn,
        "model",
        "grosshirn_candidates",
        &observed_at,
        "policy",
        &grosshirn_candidates,
    )?;

    list_resources(paths, 12)
}

pub fn record_resource_status(
    paths: &Paths,
    category: &str,
    name: &str,
    status: &str,
    detail: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    upsert_resource(&conn, category, name, &now_iso(), status, detail)?;
    Ok(())
}

pub fn list_resources(paths: &Paths, limit: usize) -> anyhow::Result<Vec<ResourceRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT category, name, observed_at, status, detail
         FROM resources
         ORDER BY category ASC, name ASC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        Ok(ResourceRecord {
            category: row.get(0)?,
            name: row.get(1)?,
            observed_at: row.get(2)?,
            status: row.get(3)?,
            detail: row.get(4)?,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn sync_skills(paths: &Paths) -> anyhow::Result<Vec<SkillRecord>> {
    let skills_dir = paths.root.join(".agents/skills");
    let now = now_iso();
    let conn = open_db(paths)?;

    if skills_dir.exists() {
        for entry in fs::read_dir(&skills_dir)? {
            let entry = entry?;
            let skill_dir = entry.path();
            let skill_md = skill_dir.join("SKILL.md");
            if !skill_md.exists() {
                continue;
            }
            let name = skill_dir
                .file_name()
                .and_then(|value| value.to_str())
                .unwrap_or_default()
                .to_string();
            let path = skill_md.display().to_string();
            let notes = fs::read_to_string(&skill_md)
                .ok()
                .map(|content| extract_skill_description(&content))
                .unwrap_or_default();
            conn.execute(
                "INSERT INTO skills(name, path, status, notes, last_seen_at)
                 VALUES(?1, ?2, 'available', ?3, ?4)
                 ON CONFLICT(name) DO UPDATE SET
                     path = excluded.path,
                     notes = excluded.notes,
                     status = excluded.status,
                     last_seen_at = excluded.last_seen_at",
                params![name, path, notes, now],
            )?;
        }
    }

    list_skills(paths)
}

fn extract_skill_description(content: &str) -> String {
    let mut lines = content.lines();
    if lines.next().map(str::trim) != Some("---") {
        return String::new();
    }

    for line in lines {
        let trimmed = line.trim();
        if trimmed == "---" {
            break;
        }
        if let Some(value) = trimmed.strip_prefix("description:") {
            return trim_skill_metadata_value(value);
        }
    }

    String::new()
}

fn trim_skill_metadata_value(value: &str) -> String {
    let trimmed = value.trim();
    if trimmed.len() >= 2 {
        let bytes = trimmed.as_bytes();
        let first = bytes[0];
        let last = bytes[trimmed.len() - 1];
        if (first == b'"' && last == b'"') || (first == b'\'' && last == b'\'') {
            return trimmed[1..trimmed.len() - 1].trim().to_string();
        }
    }
    trimmed.to_string()
}

pub fn list_skills(paths: &Paths) -> anyhow::Result<Vec<SkillRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT name, path, status, notes
         FROM skills
         ORDER BY name ASC",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok(SkillRecord {
            name: row.get(0)?,
            path: row.get(1)?,
            status: row.get(2)?,
            notes: row.get(3)?,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_homepage_revisions(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<HomepageRevisionRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT created_at, source_channel, title, headline, owner_branding_applied, notes
         FROM homepage_revisions
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        Ok(HomepageRevisionRecord {
            created_at: row.get(0)?,
            source_channel: row.get(1)?,
            title: row.get(2)?,
            headline: row.get(3)?,
            owner_branding_applied: row.get::<_, i64>(4)? != 0,
            notes: row.get(5)?,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn record_bios_upload(
    paths: &Paths,
    speaker: &str,
    source_channel: &str,
    note: &str,
    file_name: &str,
    public_path: &str,
    mime_type: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.execute(
        "INSERT INTO bios_uploads(created_at, speaker, source_channel, note, file_name, public_path, mime_type)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            now_iso(),
            speaker.trim(),
            source_channel,
            note.trim(),
            file_name,
            public_path,
            mime_type
        ],
    )?;
    Ok(())
}

pub fn list_bios_uploads(paths: &Paths, limit: usize) -> anyhow::Result<Vec<BiosUploadRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT created_at, speaker, source_channel, note, file_name, public_path, mime_type
         FROM bios_uploads
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        Ok(BiosUploadRecord {
            created_at: row.get(0)?,
            speaker: row.get(1)?,
            source_channel: row.get(2)?,
            note: row.get(3)?,
            file_name: row.get(4)?,
            public_path: row.get(5)?,
            mime_type: row.get(6)?,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn enqueue_loop_interrupt(
    paths: &Paths,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> anyhow::Result<i64> {
    let conn = open_db(paths)?;
    let created_at = now_iso();
    conn.execute(
        "INSERT INTO loop_interrupts(created_at, source_channel, speaker, message, status)
         VALUES(?1, ?2, ?3, ?4, 'pending')",
        params![created_at, source_channel, speaker.trim(), message.trim()],
    )?;
    let interrupt_id = conn.last_insert_rowid();
    let _ = record_person_interaction_with_conn(
        &conn,
        paths,
        source_channel,
        speaker,
        message,
        &format!("loop_interrupt:{interrupt_id}"),
    );
    let _ = record_agent_event_with_conn(
        &conn,
        "interrupt/queued",
        None,
        "",
        &format!(
            "Interrupt from {} via {} was recorded.",
            speaker.trim(),
            source_channel
        ),
        &serde_json::to_string(&serde_json::json!({
            "interruptId": interrupt_id,
            "speaker": speaker.trim(),
            "sourceChannel": source_channel,
            "message": message.trim(),
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    Ok(interrupt_id)
}

pub fn claim_loop_interrupt(
    paths: &Paths,
    interrupt_id: i64,
) -> anyhow::Result<Option<LoopInterruptRecord>> {
    let mut conn = open_db(paths)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let interrupt = claim_loop_interrupt_with_conn(&tx, interrupt_id)?;
    tx.commit()?;
    Ok(interrupt)
}

pub fn claim_next_loop_interrupt(paths: &Paths) -> anyhow::Result<Option<LoopInterruptRecord>> {
    let mut conn = open_db(paths)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let interrupt = claim_next_loop_interrupt_with_conn(&tx)?;
    tx.commit()?;
    Ok(interrupt)
}

pub fn complete_loop_interrupt(
    paths: &Paths,
    interrupt_id: i64,
    response: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE loop_interrupts
         SET status = 'completed',
             processed_at = ?2,
             response = ?3,
             error = NULL
         WHERE id = ?1",
        params![interrupt_id, now_iso(), response],
    )?;
    Ok(())
}

pub fn fail_loop_interrupt(paths: &Paths, interrupt_id: i64, error: &str) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE loop_interrupts
         SET status = 'failed',
             processed_at = ?2,
             error = ?3
         WHERE id = ?1",
        params![interrupt_id, now_iso(), error],
    )?;
    Ok(())
}

pub fn load_loop_interrupt_by_id(
    paths: &Paths,
    interrupt_id: i64,
) -> anyhow::Result<Option<LoopInterruptRecord>> {
    let conn = open_db(paths)?;
    load_loop_interrupt_by_id_with_conn(&conn, interrupt_id)
}

fn load_loop_interrupt_by_id_with_conn(
    conn: &Connection,
    interrupt_id: i64,
) -> anyhow::Result<Option<LoopInterruptRecord>> {
    conn.query_row(
        "SELECT id, created_at, source_channel, speaker, message, status, response, error
         FROM loop_interrupts
         WHERE id = ?1",
        params![interrupt_id],
        |row| {
            Ok(LoopInterruptRecord {
                id: row.get(0)?,
                created_at: row.get(1)?,
                source_channel: row.get(2)?,
                speaker: row.get(3)?,
                message: row.get(4)?,
                status: row.get(5)?,
                response: row.get(6)?,
                error: row.get(7)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

pub fn queue_loop_interrupt_as_task(
    paths: &Paths,
    interrupt_id: i64,
) -> anyhow::Result<Option<TaskRecord>> {
    let mut conn = open_db(paths)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let Some(interrupt) = claim_loop_interrupt_with_conn(&tx, interrupt_id)? else {
        tx.commit()?;
        return Ok(None);
    };
    let task = create_task_from_interrupt_with_conn(paths, &tx, &interrupt)?;
    tx.execute(
        "UPDATE loop_interrupts
         SET status = 'queued',
             processed_at = ?2,
             response = ?3,
             error = NULL
         WHERE id = ?1",
        params![
            interrupt_id,
            now_iso(),
            format!("queued as task {} ({})", task.id, task.title)
        ],
    )?;
    tx.commit()?;
    Ok(Some(task))
}

pub fn ingest_pending_loop_interrupts(
    paths: &Paths,
    max_items: usize,
) -> anyhow::Result<Vec<TaskRecord>> {
    let mut conn = open_db(paths)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let mut tasks = Vec::new();
    for _ in 0..max_items {
        let Some(interrupt) = claim_next_loop_interrupt_with_conn(&tx)? else {
            break;
        };
        let task = create_task_from_interrupt_with_conn(paths, &tx, &interrupt)?;
        tx.execute(
            "UPDATE loop_interrupts
             SET status = 'queued',
                 processed_at = ?2,
                 response = ?3,
                 error = NULL
             WHERE id = ?1",
            params![
                interrupt.id,
                now_iso(),
                format!("queued as task {} ({})", task.id, task.title)
            ],
        )?;
        tasks.push(task);
    }
    tx.commit()?;
    Ok(tasks)
}

pub fn list_open_tasks(paths: &Paths, limit: usize) -> anyhow::Result<Vec<TaskRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count,
                last_checkpoint_summary, last_checkpoint_at, last_output
         FROM tasks
         WHERE status IN ('queued', 'active', 'blocked', 'await_review')
         ORDER BY
             CASE status WHEN 'active' THEN 0 WHEN 'queued' THEN 1 ELSE 2 END,
             priority_score DESC,
             id ASC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_task_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_recent_completed_owner_tasks(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<TaskRecord>> {
    let owner_name = load_bios(paths).owner.name;
    let conn = open_db(paths)?;
    let fetch_limit = (limit.max(1) * 4) as i64;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count,
                last_checkpoint_summary, last_checkpoint_at, last_output
         FROM tasks
         WHERE status = 'done'
           AND source_channel IN ('bios', 'homepage', 'terminal', 'attach_terminal', 'email')
         ORDER BY updated_at DESC, id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![fetch_limit], map_task_row)?;
    Ok(rows
        .filter_map(Result::ok)
        .filter(|task| {
            task.task_kind == "owner_interrupt"
                || matches!(task.trust_level.as_str(), "owner_trust" | "owner_external")
                || is_owner_message(&task.speaker, &owner_name)
        })
        .take(limit)
        .collect())
}

pub fn list_recent_task_outcomes(
    paths: &Paths,
    exclude_task_id: i64,
    limit: usize,
) -> anyhow::Result<Vec<TaskRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count,
                last_checkpoint_summary, last_checkpoint_at, last_output
         FROM tasks
         WHERE id != ?1
           AND COALESCE(last_output, '') <> ''
           AND (
                source_channel IN ('bios', 'homepage', 'terminal', 'attach_terminal')
                OR task_kind IN ('owner_interrupt', 'channel_expansion', 'grosshirn_procurement', 'grosshirn_activation', 'model_or_resource')
           )
         ORDER BY updated_at DESC, id DESC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![exclude_task_id, limit as i64], map_task_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_queued_tasks(paths: &Paths, limit: usize) -> anyhow::Result<Vec<TaskRecord>> {
    let conn = open_db(paths)?;
    let brain_state =
        load_brain_routing_state_row(&conn)?.unwrap_or_else(default_brain_routing_state);
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count,
                last_checkpoint_summary, last_checkpoint_at, last_output
         FROM tasks
         WHERE status = 'queued'
         ORDER BY priority_score DESC, id ASC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_task_row)?;
    let mut tasks = rows
        .filter_map(Result::ok)
        .filter(|task| !task_is_in_grosshirn_cooldown_state(&brain_state, task.id))
        .collect::<Vec<_>>();
    tasks.sort_by(|left, right| {
        right
            .priority_score
            .cmp(&left.priority_score)
            .then_with(
                || match (left.task_kind.as_str(), right.task_kind.as_str()) {
                    ("owner_interrupt", "owner_interrupt") => right.id.cmp(&left.id),
                    _ => left.id.cmp(&right.id),
                },
            )
    });
    Ok(tasks)
}

pub fn load_active_task(paths: &Paths) -> anyhow::Result<Option<TaskRecord>> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count,
                last_checkpoint_summary, last_checkpoint_at, last_output
         FROM tasks
         WHERE status = 'active'
         ORDER BY priority_score DESC, id ASC
         LIMIT 1",
        [],
        map_task_row,
    )
    .optional()
    .map_err(Into::into)
}

pub fn latest_open_task_by_kind(
    paths: &Paths,
    task_kind: &str,
) -> anyhow::Result<Option<TaskRecord>> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count,
                last_checkpoint_summary, last_checkpoint_at, last_output
         FROM tasks
         WHERE task_kind = ?1
           AND status IN ('queued', 'active', 'await_review')
         ORDER BY id DESC
         LIMIT 1",
        params![task_kind],
        map_task_row,
    )
    .optional()
    .map_err(Into::into)
}

pub fn activate_selected_task(paths: &Paths, task_id: i64) -> anyhow::Result<Option<TaskRecord>> {
    if task_is_in_grosshirn_cooldown(paths, task_id) {
        return Ok(None);
    }
    clear_stale_grosshirn_boost_for_selected_task(paths, task_id)?;
    let conn = open_db(paths)?;
    let status = conn
        .query_row(
            "SELECT status FROM tasks WHERE id = ?1",
            params![task_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;

    let Some(status) = status else {
        return Ok(None);
    };
    if status != "queued" {
        return Ok(load_task_by_id(paths, task_id)?);
    }

    conn.execute(
        "UPDATE tasks
         SET status = 'active',
             updated_at = ?2,
             run_count = run_count + 1
         WHERE id = ?1",
        params![task_id, now_iso()],
    )?;
    let task = load_task_by_id(paths, task_id)?
        .ok_or_else(|| anyhow::anyhow!("task {task_id} vanished after activation"))?;
    let _ = record_agent_event(
        paths,
        "task/selected",
        Some(task.id),
        &task.title,
        &format!(
            "Task {} wurde als naechster bounded Fokus ausgewaehlt.",
            task.id
        ),
        "{}",
    );
    let selected_mode = match task.task_kind.as_str() {
        "self_preservation" => "self_preservation",
        "recovery" => "recovery",
        "worker_review" | "self_review" | "proactive_contact_review" => "review",
        _ => "execute_task",
    };
    set_focus_state(
        paths,
        selected_mode,
        Some(task.id),
        &task.title,
        &format!(
            "Selected task {} for unified {} mode.",
            task.id, selected_mode
        ),
    )?;
    Ok(Some(task))
}

pub fn select_next_task(paths: &Paths) -> anyhow::Result<Option<TaskRecord>> {
    let queued = list_queued_tasks(paths, 256)?;
    let Some(task_id) = queued
        .iter()
        .find(|task| !task_is_in_grosshirn_cooldown(paths, task.id))
        .map(|task| task.id)
    else {
        set_focus_state(paths, "idle", None, "", "No queued tasks available.")?;
        return Ok(None);
    };

    clear_stale_grosshirn_boost_for_selected_task(paths, task_id)?;
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'active',
             updated_at = ?2,
             run_count = run_count + 1
         WHERE id = ?1",
        params![task_id, now_iso()],
    )?;
    let task = load_task_by_id(paths, task_id)?
        .ok_or_else(|| anyhow::anyhow!("task {task_id} vanished after activation"))?;
    let _ = record_agent_event(
        paths,
        "task/selected",
        Some(task.id),
        &task.title,
        &format!(
            "Task {} wurde als naechster bounded Fokus ausgewaehlt.",
            task.id
        ),
        "{}",
    );
    let selected_mode = match task.task_kind.as_str() {
        "self_preservation" => "self_preservation",
        "recovery" => "recovery",
        "worker_review" | "self_review" | "proactive_contact_review" => "review",
        _ => "execute_task",
    };
    set_focus_state(
        paths,
        selected_mode,
        Some(task.id),
        &task.title,
        &format!(
            "Selected task {} for unified {} mode.",
            task.id, selected_mode
        ),
    )?;
    Ok(Some(task))
}

pub fn record_task_checkpoint(
    paths: &Paths,
    task_id: i64,
    checkpoint_kind: &str,
    summary: &str,
    detail: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    let created_at = now_iso();
    conn.execute(
        "INSERT INTO task_checkpoints(task_id, created_at, checkpoint_kind, summary, detail)
         VALUES(?1, ?2, ?3, ?4, ?5)",
        params![task_id, created_at, checkpoint_kind, summary, detail],
    )?;
    conn.execute(
        "UPDATE tasks
         SET updated_at = ?2,
             last_checkpoint_summary = ?3,
             last_checkpoint_at = ?4
         WHERE id = ?1",
        params![task_id, now_iso(), summary, created_at],
    )?;
    Ok(())
}

pub fn list_task_checkpoints(
    paths: &Paths,
    task_id: i64,
    limit: usize,
) -> anyhow::Result<Vec<TaskCheckpointRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT task_id, created_at, checkpoint_kind, summary, detail
         FROM task_checkpoints
         WHERE task_id = ?1
         ORDER BY id DESC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![task_id, limit as i64], |row| {
        Ok(TaskCheckpointRecord {
            task_id: row.get(0)?,
            created_at: row.get(1)?,
            checkpoint_kind: row.get(2)?,
            summary: row.get(3)?,
            detail: row.get(4)?,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn delegate_task_to_worker(
    paths: &Paths,
    task: &TaskRecord,
    worker_kind: &str,
    contract_title: &str,
    contract_detail: &str,
    request_note: &str,
    output: Option<&str>,
) -> anyhow::Result<WorkerJobRecord> {
    record_task_checkpoint(paths, task.id, "delegate", contract_title, contract_detail)?;
    let conn = open_db(paths)?;
    let timestamp = now_iso();
    conn.execute(
        "UPDATE tasks
         SET status = 'await_review',
             updated_at = ?2,
             last_output = ?3
         WHERE id = ?1",
        params![task.id, timestamp, output.unwrap_or("")],
    )?;
    conn.execute(
        "INSERT INTO worker_jobs(
            created_at, updated_at, parent_task_id, parent_task_title, worker_kind,
            contract_title, contract_detail, status, request_note
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, 'queued', ?8)",
        params![
            timestamp,
            timestamp,
            task.id,
            task.title,
            worker_kind,
            contract_title,
            contract_detail,
            request_note
        ],
    )?;
    let worker_job_id = conn.last_insert_rowid();
    conn.execute(
        "UPDATE tasks SET worker_job_id = ?2 WHERE id = ?1",
        params![task.id, worker_job_id],
    )?;
    let _ = record_agent_event(
        paths,
        "task/delegated",
        Some(task.id),
        &task.title,
        &format!(
            "Task {} wurde an Worker {} delegiert.",
            task.id, worker_kind
        ),
        &serde_json::to_string(&serde_json::json!({
            "workerJobId": worker_job_id,
            "workerKind": worker_kind,
            "contractTitle": contract_title,
            "requestNote": request_note,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    load_worker_job_by_id(paths, worker_job_id)?
        .ok_or_else(|| anyhow::anyhow!("failed to reload delegated worker job {worker_job_id}"))
}

pub fn requeue_task(
    paths: &Paths,
    task_id: i64,
    summary: &str,
    detail: &str,
    output: Option<&str>,
) -> anyhow::Result<()> {
    record_task_checkpoint(paths, task_id, "continue", summary, detail)?;
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'queued',
             updated_at = ?2,
             last_output = ?3
         WHERE id = ?1",
        params![task_id, now_iso(), output.unwrap_or("")],
    )?;
    let title = load_task_by_id(paths, task_id)?
        .map(|task| task.title)
        .unwrap_or_default();
    let _ = record_agent_event(
        paths,
        "task/requeued",
        Some(task_id),
        &title,
        summary,
        &serde_json::to_string(&serde_json::json!({
            "detail": detail,
            "output": output.unwrap_or(""),
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    set_focus_state(
        paths,
        "reprioritize",
        None,
        "",
        &format!("Task {} checkpointed and requeued.", task_id),
    )?;
    Ok(())
}

pub fn continue_active_task(
    paths: &Paths,
    task_id: i64,
    next_mode: &str,
    summary: &str,
    detail: &str,
    output: Option<&str>,
) -> anyhow::Result<()> {
    record_task_checkpoint(paths, task_id, "continue", summary, detail)?;
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'active',
             updated_at = ?2,
             last_output = ?3,
             run_count = run_count + 1
         WHERE id = ?1",
        params![task_id, now_iso(), output.unwrap_or("")],
    )?;
    let task = load_task_by_id(paths, task_id)?
        .ok_or_else(|| anyhow::anyhow!("task {task_id} vanished after continue_active_task"))?;
    let _ = record_agent_event(
        paths,
        "task/continued",
        Some(task_id),
        &task.title,
        summary,
        &serde_json::to_string(&serde_json::json!({
            "detail": detail,
            "output": output.unwrap_or(""),
            "nextMode": next_mode,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    set_focus_state(
        paths,
        next_mode,
        Some(task_id),
        &task.title,
        &format!(
            "Task {} stayed active in unified {} mode after a productive bounded step.",
            task_id, next_mode
        ),
    )?;
    Ok(())
}

pub fn yield_active_task_for_preemption(
    paths: &Paths,
    task_id: i64,
    reason: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'queued',
             updated_at = ?2
         WHERE id = ?1
           AND status = 'active'",
        params![task_id, now_iso()],
    )?;
    let title = load_task_by_id(paths, task_id)?
        .map(|task| task.title)
        .unwrap_or_default();
    let _ = record_agent_event(paths, "task/yielded", Some(task_id), &title, reason, "{}");
    Ok(())
}

pub fn requeue_task_with_checkpoint_kind(
    paths: &Paths,
    task_id: i64,
    checkpoint_kind: &str,
    summary: &str,
    detail: &str,
    output: Option<&str>,
) -> anyhow::Result<()> {
    record_task_checkpoint(paths, task_id, checkpoint_kind, summary, detail)?;
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'queued',
             updated_at = ?2,
             last_output = ?3,
             run_count = CASE
                 WHEN ?4 = 'context_preparation_ready' THEN 0
                 ELSE run_count
             END
         WHERE id = ?1",
        params![task_id, now_iso(), output.unwrap_or(""), checkpoint_kind],
    )?;
    let title = load_task_by_id(paths, task_id)?
        .map(|task| task.title)
        .unwrap_or_default();
    let _ = record_agent_event(
        paths,
        "task/requeued",
        Some(task_id),
        &title,
        summary,
        &serde_json::to_string(&serde_json::json!({
            "checkpointKind": checkpoint_kind,
            "detail": detail,
            "output": output.unwrap_or(""),
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    set_focus_state(
        paths,
        "reprioritize",
        None,
        "",
        &format!(
            "Task {} checkpointed as {} and requeued.",
            task_id, checkpoint_kind
        ),
    )?;
    Ok(())
}

pub fn complete_review_task(
    paths: &Paths,
    review_task: &TaskRecord,
    task_status: &str,
    summary: &str,
    detail: &str,
    output: Option<&str>,
) -> anyhow::Result<()> {
    let Some(parent_task_id) = review_task.parent_task_id else {
        return complete_task(paths, review_task.id, summary, detail, output);
    };
    let worker_job_id = review_task.worker_job_id;

    record_task_checkpoint(paths, review_task.id, "done", summary, detail)?;
    let conn = open_db(paths)?;
    let timestamp = now_iso();
    conn.execute(
        "UPDATE tasks
         SET status = 'done',
             updated_at = ?2,
             last_output = ?3
         WHERE id = ?1",
        params![review_task.id, timestamp, output.unwrap_or("")],
    )?;
    if let Some(worker_job_id) = worker_job_id {
        conn.execute(
            "UPDATE worker_jobs
             SET status = 'reviewed',
                 updated_at = ?2,
                 review_summary = ?3,
                 review_detail = ?4,
                 completed_at = ?5
             WHERE id = ?1",
            params![worker_job_id, timestamp, summary, detail, timestamp],
        )?;
    }

    let parent_task = load_task_by_id(paths, parent_task_id)?.ok_or_else(|| {
        anyhow::anyhow!("failed to reload parent task {parent_task_id} for review")
    })?;

    match task_status {
        "continue" => {
            record_task_checkpoint(paths, parent_task_id, "review_continue", summary, detail)?;
            conn.execute(
                "UPDATE tasks
                 SET status = 'queued',
                     updated_at = ?2,
                     last_output = ?3
                 WHERE id = ?1",
                params![parent_task_id, timestamp, output.unwrap_or("")],
            )?;
            set_focus_state(
                paths,
                "reprioritize",
                Some(parent_task_id),
                &review_task.title,
                &format!(
                    "Worker review for task {} requested further bounded work.",
                    parent_task_id
                ),
            )?;
            let _ = record_agent_event(
                paths,
                "review/completed",
                Some(parent_task_id),
                &review_task.title,
                "Worker-Review fordert weitere bounded Arbeit an.",
                "{}",
            );
            let _ = reject_learning_candidates_for_task(paths, parent_task_id, "review_continue");
        }
        "blocked" => {
            record_task_checkpoint(paths, parent_task_id, "review_blocked", summary, detail)?;
            conn.execute(
                "UPDATE tasks
                 SET status = 'blocked',
                     updated_at = ?2,
                     last_output = ?3
                 WHERE id = ?1",
                params![parent_task_id, timestamp, output.unwrap_or("")],
            )?;
            set_focus_state(
                paths,
                "blocked",
                Some(parent_task_id),
                &review_task.title,
                &format!("Worker review blocked parent task {}.", parent_task_id),
            )?;
            let _ = record_agent_event(
                paths,
                "review/completed",
                Some(parent_task_id),
                &review_task.title,
                "Worker review blocks the parent task.",
                "{}",
            );
            let _ = release_task_grosshirn_boost(
                paths,
                parent_task_id,
                "The task was blocked despite the temporary grosshirn boost and falls back to kleinhirn.",
            );
            let _ = reject_learning_candidates_for_task(paths, parent_task_id, "review_blocked");
        }
        _ => {
            if let Some(gate_failure) =
                task_completion_gate_failure(paths, &parent_task, Some(summary), Some(detail))
            {
                let gated_detail =
                    format!("{detail}\n\nThe review gate refused completion:\n{gate_failure}");
                record_task_checkpoint(
                    paths,
                    parent_task_id,
                    "review_continue",
                    summary,
                    &gated_detail,
                )?;
                conn.execute(
                    "UPDATE tasks
                     SET status = 'queued',
                         updated_at = ?2,
                         last_output = ?3
                     WHERE id = ?1",
                    params![parent_task_id, timestamp, output.unwrap_or("")],
                )?;
                set_focus_state(
                    paths,
                    "reprioritize",
                    Some(parent_task_id),
                    &parent_task.title,
                    &format!("The review gate reopened task {}.", parent_task_id),
                )?;
                let _ = record_agent_event(
                    paths,
                    "review/reopened",
                    Some(parent_task_id),
                    &review_task.title,
                    &gate_failure,
                    "{}",
                );
                if tracks_progress_journal(&parent_task.task_kind) {
                    let _ = record_progress_journal(
                        paths,
                        &format!(
                            "Task #{} {} is not yet proven as an improvement.",
                            parent_task.id, parent_task.title
                        ),
                        &gated_detail,
                        "review_reopened",
                    );
                }
                let _ =
                    reject_learning_candidates_for_task(paths, parent_task_id, "review_reopened");
                return Ok(());
            }
            let _ = release_task_grosshirn_boost(
                paths,
                parent_task_id,
                "The task is complete; the temporary grosshirn boost is being released back to kleinhirn.",
            );
            record_task_checkpoint(paths, parent_task_id, "review_done", summary, detail)?;
            conn.execute(
                "UPDATE tasks
                 SET status = 'done',
                     updated_at = ?2,
                     last_output = ?3
                 WHERE id = ?1",
                params![parent_task_id, timestamp, output.unwrap_or("")],
            )?;
            conn.execute(
                "UPDATE focus_state
                 SET last_task_completed_at = ?1
                 WHERE singleton = 1",
                params![timestamp],
            )?;
            set_focus_state(
                paths,
                "review",
                Some(parent_task_id),
                &review_task.title,
                &format!(
                    "Worker review accepted and parent task {} is complete.",
                    parent_task_id
                ),
            )?;
            let _ = record_agent_event(
                paths,
                "review/completed",
                Some(parent_task_id),
                &review_task.title,
                "Worker review was accepted and the parent task was completed.",
                "{}",
            );
            if tracks_progress_journal(&parent_task.task_kind) {
                let _ = record_progress_journal(
                    paths,
                    &format!(
                        "Task #{} {} was accepted as an effective improvement.",
                        parent_task.id, parent_task.title
                    ),
                    detail,
                    "review_completed",
                );
            }
            let _ = promote_learning_candidates_for_task(paths, parent_task_id, "review_completed");
        }
    }

    Ok(())
}

pub fn recover_orphaned_review_waits(paths: &Paths) -> anyhow::Result<Vec<i64>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id
         FROM tasks AS parent
         WHERE parent.status = 'await_review'
           AND parent.task_kind NOT IN ('self_review', 'worker_review', 'proactive_contact_review')
           AND NOT EXISTS(
                SELECT 1
                FROM tasks AS child
                WHERE child.parent_task_id = parent.id
                  AND child.task_kind IN ('self_review', 'worker_review', 'proactive_contact_review')
                  AND child.status IN ('queued', 'active', 'await_review')
           )
         ORDER BY parent.priority_score DESC, parent.id ASC",
    )?;
    let parent_ids = stmt
        .query_map([], |row| row.get::<_, i64>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);

    let timestamp = now_iso();
    let mut recovered = Vec::new();
    for parent_id in parent_ids {
        let Some(parent_task) = load_task_by_id(paths, parent_id)? else {
            continue;
        };
        if parent_task.status != "await_review" {
            continue;
        }
        let summary = format!(
            "Review wait recovered for task #{} {}.",
            parent_task.id, parent_task.title
        );
        let detail = "The parent task was still waiting for review, but no queued/active review task remained. The kernel reopened the task so work can continue instead of staying stranded in await_review.";
        record_task_checkpoint(paths, parent_task.id, "review_recovered", &summary, detail)?;
        conn.execute(
            "UPDATE tasks
             SET status = 'queued',
                 updated_at = ?2
             WHERE id = ?1",
            params![parent_task.id, timestamp],
        )?;
        set_focus_state(
            paths,
            "reprioritize",
            Some(parent_task.id),
            &parent_task.title,
            "A stranded await_review parent was reopened because no live review task remained.",
        )?;
        let _ = record_agent_event(
            paths,
            "review/recovered",
            Some(parent_task.id),
            &parent_task.title,
            "The parent task was reopened after the review child disappeared.",
            "{}",
        );
        recovered.push(parent_task.id);
    }

    Ok(recovered)
}

fn owner_interrupt_demands_real_host_change(task: &TaskRecord) -> bool {
    let lowered = format!("{} {}", task.title, task.detail).to_lowercase();
    contains_any(
        &lowered,
        &[
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
        ],
    )
}

fn owner_interrupt_has_real_execution_evidence(
    task: &TaskRecord,
    review_summary: Option<&str>,
    review_detail: Option<&str>,
) -> bool {
    let evidence = [
        task.last_checkpoint_summary.as_deref().unwrap_or(""),
        task.last_output.as_deref().unwrap_or(""),
        review_summary.unwrap_or(""),
        review_detail.unwrap_or(""),
    ]
    .join("\n")
    .to_lowercase();
    if evidence.trim().is_empty() {
        return false;
    }
    let unresolved = contains_any(
        &evidence,
        &[
            "need targeted history",
            "need more context",
            "need exact",
            "i need",
            "missing context",
            "lack context",
            "bevor",
            "before issuing",
            "unknown",
            "unclear",
            "keine analyse statt umsetzung",
            "analyse statt umsetzung",
            "kontext fehlt",
            "kontext",
            "history/state",
        ],
    );
    if unresolved {
        return false;
    }
    let action = contains_any(
        &evidence,
        &[
            "umgestellt",
            "gestellt",
            "geändert",
            "geaendert",
            "changed",
            "set ",
            "gesetzt",
            "configured",
            "konfiguriert",
            "verified",
            "verifiziert",
            "setxkbmap",
            "localectl",
            "kwriteconfig",
            "qdbus",
            "xkb",
        ],
    );
    let target = contains_any(
        &evidence,
        &[
            "keyboard",
            "tastatur",
            "layout",
            "deutsch",
            "german",
            "de/de-latin1",
            "de ",
            "layout=de",
            "xkb",
        ],
    );
    action && target
}

fn task_completion_gate_failure(
    paths: &Paths,
    task: &TaskRecord,
    review_summary: Option<&str>,
    review_detail: Option<&str>,
) -> Option<String> {
    let bios = load_bios(paths);
    let organigram = load_organigram(paths);
    let root_auth = load_root_auth(paths);
    let homepage = load_homepage_policy(paths);
    let trust = load_owner_trust(paths).unwrap_or_default();
    let census = load_census(paths);
    let mut failures = Vec::new();

    match task.task_kind.as_str() {
        "homepage_bridge" => {
            if !homepage.homepage_ready {
                failures.push("Homepage is not marked as ready yet.");
            }
            if !homepage.bios_visible {
                failures.push("BIOS is not visible enough on the homepage yet.");
            }
            let browser_artifacts_ready = paths.root.join("runtime/browser");
            let browser_verified = fs::read_dir(&browser_artifacts_ready)
                .ok()
                .map(|mut entries| entries.next().is_some())
                .unwrap_or(false);
            if !browser_verified {
                failures.push(
                    "Homepage or BIOS work has not been verified yet with real browser work.",
                );
            }
        }
        "root_trust" => {
            if !root_auth.configured {
                failures.push("Superpassword is not set yet.");
            }
        }
        "organigram_contract" => {
            if organigram.owner.name.trim().is_empty() {
                failures.push("Owner is still missing from the organigram.");
            }
            if organigram.reports_to.trim().is_empty()
                && organigram.ceo.trim().is_empty()
                && organigram.board.is_empty()
            {
                failures.push("CEO, board, or reports-to is still missing from the organigram.");
            }
        }
        "bios_contract" => {
            if !bios.presented_on_web || bios.website_path.trim().is_empty() {
                failures.push("BIOS is not being presented cleanly on the website yet.");
            }
        }
        "owner_interrupt" => {
            if owner_interrupt_demands_real_host_change(task)
                && !owner_interrupt_has_real_execution_evidence(task, review_summary, review_detail)
            {
                failures.push(
                    "A direct owner instruction with a real host mutation must not count as complete without strong execution and verification evidence.",
                );
            }
        }
        "owner_binding" => {
            if !trust.owner_contact_established {
                failures.push("There is still no reliable owner contact.");
            }
            if !trust.bios_primary_channel_confirmed {
                failures.push(
                    "BIOS communication has not yet been taken over as the primary trust path.",
                );
            }
        }
        "bios_freeze" => {
            if !root_auth.configured {
                failures.push("BIOS may not count as complete without a configured superpassword.");
            }
            if !trust.bios_primary_channel_confirmed {
                failures
                    .push("BIOS may not be frozen without an established BIOS communication path.");
            }
            if !bios.frozen {
                failures.push("BIOS is not frozen yet.");
            }
        }
        "owner_branding" => {
            if !homepage.owner_branding_locked {
                failures.push("Owner branding is not locked yet.");
            }
        }
        "resource_census" => {
            if census.cpu_threads.unwrap_or(0) == 0 || census.total_memory_gb.unwrap_or(0) == 0 {
                failures.push("A reliable system census is not available yet.");
            }
        }
        "model_or_resource" => {
            if crate::brain_runtime::local_kleinhirn_upgrade_available(paths) {
                failures.push("A stronger local kleinhirn is already viable, but not active yet.");
            }
        }
        "grosshirn_procurement" => {
            let grosshirn_configured = crate::brain_runtime::grosshirn_runtime_configured(paths);
            if trust.brain_access_mode == "kleinhirn_plus_grosshirn" || grosshirn_configured {
                failures.push(
                    "Grosshirn is already enabled or configured; procurement is no longer an open need.",
                );
            } else {
                failures.push("There is still no approved or configured grosshirn access.");
            }
        }
        "grosshirn_activation" => {
            let grosshirn_configured = crate::brain_runtime::grosshirn_runtime_configured(paths);
            if trust.brain_access_mode == "kleinhirn_plus_grosshirn" && grosshirn_configured {
                failures.push("Grosshirn mode is already enabled and configured.");
            }
            if trust.brain_access_mode != "kleinhirn_plus_grosshirn" {
                failures.push("Brain access is not set to `kleinhirn_plus_grosshirn` yet.");
            }
            if !grosshirn_configured {
                failures.push("Grosshirn credentials or runtime configuration are still missing.");
            }
        }
        _ => {}
    }

    if failures.is_empty() {
        None
    } else {
        Some(failures.join(" "))
    }
}

fn tracks_progress_journal(task_kind: &str) -> bool {
    matches!(
        task_kind,
        "environment_discovery"
            | "tool_exploration"
            | "progress_reflection"
            | "person_relationship_review"
            | "homepage_bridge"
            | "model_or_resource"
            | "grosshirn_procurement"
            | "grosshirn_activation"
    )
}

pub fn complete_task(
    paths: &Paths,
    task_id: i64,
    summary: &str,
    detail: &str,
    output: Option<&str>,
) -> anyhow::Result<()> {
    let task = load_task_by_id(paths, task_id)?;
    record_task_checkpoint(paths, task_id, "done", summary, detail)?;
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'done',
             updated_at = ?2,
             last_output = ?3
         WHERE id = ?1",
        params![task_id, now_iso(), output.unwrap_or("")],
    )?;
    let title = task
        .as_ref()
        .map(|task| task.title.clone())
        .unwrap_or_default();
    let _ = record_agent_event(
        paths,
        "task/completed",
        Some(task_id),
        &title,
        summary,
        &serde_json::to_string(&serde_json::json!({
            "detail": detail,
            "output": output.unwrap_or(""),
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    conn.execute(
        "UPDATE focus_state
         SET last_task_completed_at = ?1
         WHERE singleton = 1",
        params![now_iso()],
    )?;
    set_focus_state(
        paths,
        "review",
        None,
        "",
        &format!("Task {} completed and awaits unified review.", task_id),
    )?;
    if let Some(task) = task.as_ref() {
        if matches!(task.task_kind.as_str(), "self_preservation" | "recovery") {
            resolve_loop_incidents_for_task(
                paths,
                task.id,
                &format!("Special task '{}' completed: {}", task.title, summary),
            )?;
            touch_loop_failure_state(
                paths,
                task.task_kind == "recovery",
                &format!("{} completed successfully.", task.task_kind),
                true,
            )?;
        }
    }
    Ok(())
}

pub fn advance_worker_jobs(
    paths: &Paths,
    max_items: usize,
) -> anyhow::Result<Vec<WorkerJobRecord>> {
    let mut completed = Vec::new();
    for _ in 0..max_items {
        let Some(job) = claim_next_worker_job(paths)? else {
            break;
        };
        let result_summary = format!(
            "Worker {} completed delegated contract: {}",
            job.worker_kind, job.contract_title
        );
        let result_detail = format!(
            "The delegated worker processed the contract.\n\nContract title: {}\nContract detail: {}\nRequest: {}\nReturn: Please check whether the direction is correct and whether more bounded work or further delegation is needed.",
            job.contract_title, job.contract_detail, job.request_note
        );
        let review_task = emit_worker_review_task(paths, &job, &result_summary, &result_detail)?;
        let updated = finalize_worker_job_for_review(
            paths,
            job.id,
            &result_summary,
            &result_detail,
            review_task.id,
        )?;
        let _ = record_agent_event(
            paths,
            "worker/reviewQueued",
            Some(updated.parent_task_id),
            &updated.parent_task_title,
            &format!(
                "Worker-Job {} hat Review-Task #{} erzeugt.",
                updated.id, review_task.id
            ),
            &serde_json::to_string(&serde_json::json!({
                "workerJobId": updated.id,
                "reviewTaskId": review_task.id,
                "workerKind": updated.worker_kind,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );
        completed.push(updated);
    }
    Ok(completed)
}

pub fn block_task(
    paths: &Paths,
    task_id: i64,
    summary: &str,
    detail: &str,
    output: Option<&str>,
) -> anyhow::Result<()> {
    let task = load_task_by_id(paths, task_id)?;
    record_task_checkpoint(paths, task_id, "blocked", summary, detail)?;
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'blocked',
             updated_at = ?2,
             last_output = ?3
         WHERE id = ?1",
        params![task_id, now_iso(), output.unwrap_or("")],
    )?;
    let title = task
        .as_ref()
        .map(|task| task.title.clone())
        .unwrap_or_default();
    let _ = record_agent_event(
        paths,
        "task/blocked",
        Some(task_id),
        &title,
        summary,
        &serde_json::to_string(&serde_json::json!({
            "detail": detail,
            "output": output.unwrap_or(""),
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    set_focus_state(
        paths,
        "blocked",
        None,
        "",
        &format!("Task {} blocked and awaits a future signal.", task_id),
    )?;
    if let Some(task) = task.as_ref()
        && matches!(task.task_kind.as_str(), "self_preservation" | "recovery")
    {
        touch_loop_failure_state(
            paths,
            task.task_kind == "recovery",
            &format!("{} task blocked: {}", task.task_kind, summary),
            false,
        )?;
    }
    Ok(())
}

pub fn reprioritize_tasks(paths: &Paths) -> anyhow::Result<()> {
    let mut active_focus: Option<(i64, String, String)> = None;
    {
        let conn = open_db(paths)?;
        let bios = load_bios(paths);
        let owner_name = bios.owner.name;
        let loop_safety = load_loop_safety_policy(paths);
        let tasks: Vec<(i64, Option<i64>, String, String, String, String)> = {
            let mut task_stmt = conn.prepare(
                "SELECT id, parent_task_id, task_kind, source_channel, speaker, detail
                 FROM tasks
                 WHERE status IN ('queued', 'active')",
            )?;
            let task_rows = task_stmt.query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, Option<i64>>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                    row.get::<_, String>(5)?,
                ))
            })?;
            task_rows.collect::<rusqlite::Result<Vec<_>>>()?
        };
        for task in tasks {
            let mut new_priority = compute_priority_score(
                &loop_safety,
                &owner_name,
                &task.2,
                &task.3,
                &task.4,
                &task.5,
            );
            if task.2 == "context_preparation"
                && let Some(parent_task_id) = task.1
            {
                let parent_priority = conn
                    .query_row(
                        "SELECT priority_score FROM tasks WHERE id = ?1",
                        params![parent_task_id],
                        |row| row.get::<_, i64>(0),
                    )
                    .optional()?;
                if let Some(parent_priority) = parent_priority {
                    new_priority = new_priority.max(parent_priority.saturating_add(5));
                }
            }
            conn.execute(
                "UPDATE tasks SET priority_score = ?2, updated_at = ?3 WHERE id = ?1",
                params![task.0, new_priority, now_iso()],
            )?;
        }
        active_focus = conn
            .query_row(
                "SELECT id, task_kind, title
                 FROM tasks
                 WHERE status = 'active'
                 ORDER BY updated_at DESC, id DESC
                 LIMIT 1",
                [],
                |row| {
                    Ok((
                        row.get::<_, i64>(0)?,
                        row.get::<_, String>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                },
            )
            .optional()?;
    }
    if let Some((task_id, task_kind, title)) = active_focus {
        let selected_mode = match task_kind.as_str() {
            "self_preservation" => "self_preservation",
            "recovery" => "recovery",
            "worker_review" | "self_review" | "proactive_contact_review" => "review",
            _ => "execute_task",
        };
        set_focus_state(
            paths,
            selected_mode,
            Some(task_id),
            &title,
            "Task priorities recalculated while the active task remains in progress.",
        )?;
    } else {
        set_focus_state(
            paths,
            "reprioritize",
            None,
            "",
            "Task priorities recalculated.",
        )?;
    }
    Ok(())
}

pub fn load_focus_state(paths: &Paths) -> anyhow::Result<FocusStateRecord> {
    let conn = open_db(paths)?;
    let queue_depth: i64 = conn.query_row(
        "SELECT COUNT(*) FROM tasks WHERE status IN ('queued', 'active')",
        [],
        |row| row.get(0),
    )?;
    conn.query_row(
        "SELECT mode, active_task_id, active_task_title, queue_depth, last_reprioritized_at, last_task_completed_at, note
         FROM focus_state
         WHERE singleton = 1",
        [],
        |row| {
            Ok(FocusStateRecord {
                mode: row.get(0)?,
                active_task_id: row.get(1)?,
                active_task_title: row.get(2)?,
                queue_depth,
                last_reprioritized_at: row.get(4)?,
                last_task_completed_at: row.get(5)?,
                note: row.get(6)?,
            })
        },
    )
    .map_err(Into::into)
}

pub fn list_worker_jobs(paths: &Paths, limit: usize) -> anyhow::Result<Vec<WorkerJobRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, parent_task_id, parent_task_title, worker_kind,
                contract_title, contract_detail, status, request_note, result_summary,
                result_detail, review_summary, review_detail, review_task_id, completed_at
         FROM worker_jobs
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_worker_job_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn record_context_package(
    paths: &Paths,
    task_id: i64,
    task_title: &str,
    context_mode: &str,
    budget_hint: usize,
    rationale: &str,
    package_json: &str,
) -> anyhow::Result<i64> {
    let conn = open_db(paths)?;
    conn.execute(
        "INSERT INTO context_packages(created_at, task_id, task_title, context_mode, budget_hint, rationale, package_json)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            now_iso(),
            task_id,
            task_title,
            context_mode,
            budget_hint as i64,
            rationale,
            package_json
        ],
    )?;
    let context_package_id = conn.last_insert_rowid();
    let _ = record_agent_event(
        paths,
        "context/prepared",
        Some(task_id),
        task_title,
        rationale,
        &serde_json::to_string(&serde_json::json!({
            "contextPackageId": context_package_id,
            "contextMode": context_mode,
            "budgetHint": budget_hint,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    Ok(context_package_id)
}

pub fn set_agent_mode(
    paths: &Paths,
    mode: &str,
    active_task_id: Option<i64>,
    active_task_title: &str,
    note: &str,
) -> anyhow::Result<()> {
    set_focus_state(paths, mode, active_task_id, active_task_title, note)
}

pub fn list_mode_transitions(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<ModeTransitionRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT created_at, from_mode, to_mode, active_task_id, active_task_title, trigger, note
         FROM mode_transitions
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], |row| {
        Ok(ModeTransitionRecord {
            created_at: row.get(0)?,
            from_mode: row.get(1)?,
            to_mode: row.get(2)?,
            active_task_id: row.get(3)?,
            active_task_title: row.get(4)?,
            trigger: row.get(5)?,
            note: row.get(6)?,
        })
    })?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_recent_agent_events(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<AgentEventRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, method, active_task_id, active_task_title, body, payload_json
         FROM agent_events
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_agent_event_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_agent_events_since(
    paths: &Paths,
    after_id: i64,
    limit: usize,
) -> anyhow::Result<Vec<AgentEventRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, method, active_task_id, active_task_title, body, payload_json
         FROM agent_events
         WHERE id > ?1
         ORDER BY id ASC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![after_id, limit as i64], map_agent_event_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_recent_loop_incidents(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<LoopIncidentRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, incident_key, severity, status, summary, detail,
                related_task_id, related_turn_id, self_preservation_task_id, hard_reset_required,
                hard_reset_report_path, resolved_at
         FROM loop_incidents
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_loop_incident_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn has_open_loop_incident(paths: &Paths, incident_key: &str) -> anyhow::Result<bool> {
    let conn = open_db(paths)?;
    let count = conn.query_row(
        "SELECT COUNT(*)
         FROM loop_incidents
         WHERE incident_key = ?1
           AND status = 'open'",
        params![incident_key],
        |row| row.get::<_, i64>(0),
    )?;
    Ok(count > 0)
}

fn summary_looks_like_retryable_runtime_stall(summary: &str) -> bool {
    let lowered = summary.trim().to_lowercase();
    lowered.contains("failed to connect to 127.0.0.1")
        || lowered.contains("connection refused")
        || lowered.contains("tcp connect error")
        || lowered.contains("os error 111")
        || lowered
            .contains("workspace execution contract is active and no machine path ran in this turn")
}

pub fn requeue_retryable_blocked_tasks_after_runtime_stall(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<i64>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count,
                last_checkpoint_summary, last_checkpoint_at, last_output
         FROM tasks
         WHERE status = 'blocked'
           AND (
                lower(COALESCE(last_checkpoint_summary, '')) LIKE '%failed to connect to 127.0.0.1%'
                OR lower(COALESCE(last_checkpoint_summary, '')) LIKE '%connection refused%'
                OR lower(COALESCE(last_checkpoint_summary, '')) LIKE '%tcp connect error%'
                OR lower(COALESCE(last_checkpoint_summary, '')) LIKE '%os error 111%'
                OR lower(COALESCE(last_checkpoint_summary, '')) LIKE '%workspace execution contract is active and no machine path ran in this turn%'
           )
         ORDER BY priority_score DESC, id ASC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_task_row)?;
    let tasks = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    drop(stmt);
    drop(conn);
    let mut revived = Vec::new();
    for task in tasks {
        let prior_summary = task.last_checkpoint_summary.clone().unwrap_or_default();
        let detail = if prior_summary.is_empty() {
            "The task had been blocked by a retryable runtime stall and is requeued for continuation."
                .to_string()
        } else {
            format!(
                "The task had been blocked by a retryable runtime stall and is requeued for continuation.\n\nPrevious blocked summary:\n{}",
                prior_summary
            )
        };
        requeue_task(
            paths,
            task.id,
            "Retryable runtime stall cleared enough to retry this task.",
            &detail,
            task.last_output.as_deref(),
        )?;
        revived.push(task.id);
    }
    Ok(revived)
}

pub fn write_pending_hard_reset_report(
    paths: &Paths,
    trigger: &str,
    reason: &str,
) -> anyhow::Result<String> {
    fs::create_dir_all(&paths.recovery_dir)?;
    let report = serde_json::json!({
        "createdAt": now_iso(),
        "trigger": trigger,
        "reason": reason,
        "agentState": load_agent_state(paths),
        "agentThread": load_agent_thread(paths).ok(),
        "focusState": load_focus_state(paths).ok(),
        "activeTurn": load_active_agent_turn(paths).ok().flatten(),
        "latestCompletedTurn": load_latest_completed_agent_turn(paths).ok().flatten(),
        "openTasks": list_open_tasks(paths, 12).unwrap_or_default(),
        "recentEvents": list_recent_agent_events(paths, 20).unwrap_or_default(),
        "recentSignals": list_recent_turn_signals(paths, 12).unwrap_or_default(),
    });
    let encoded = serde_json::to_vec_pretty(&report)?;
    fs::write(&paths.pending_hard_reset_report_path, encoded)?;
    let report_path = paths.pending_hard_reset_report_path.display().to_string();
    let _ = record_agent_event(
        paths,
        "loop/hardResetReportWritten",
        None,
        "",
        "A hard-reset debug report was written for the next restart.",
        &serde_json::to_string(&serde_json::json!({
            "trigger": trigger,
            "reason": reason,
            "reportPath": report_path,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    Ok(report_path)
}

pub fn activate_startup_recovery(paths: &Paths) -> anyhow::Result<Option<TaskRecord>> {
    if !paths.pending_hard_reset_report_path.exists() {
        return Ok(None);
    }

    let report_path = paths.pending_hard_reset_report_path.display().to_string();
    let report_text = match fs::read_to_string(&paths.pending_hard_reset_report_path) {
        Ok(text) => text,
        Err(err) => serde_json::to_string_pretty(&serde_json::json!({
            "summary": "Pending hard reset report could not be read cleanly.",
            "detail": err.to_string(),
            "path": report_path,
        }))
        .unwrap_or_else(|_| {
            "{\"summary\":\"failed to read pending hard reset report\"}".to_string()
        }),
    };

    let summary = serde_json::from_str::<serde_json::Value>(&report_text)
        .ok()
        .and_then(|value| {
            value
                .get("reason")
                .and_then(|item| item.as_str())
                .map(ToString::to_string)
        })
        .unwrap_or_else(|| "Automatic restart with debug report detected.".to_string());

    let task = enqueue_internal_task(
        paths,
        None,
        "recovery",
        "Run Infinity Loop hard-reset recovery",
        &format!(
            "Analyze the hard-reset debug report at {} and derive bounded recovery work from it. Stabilize the Infinity Loop, recut context, and only then return to reprioritize.\n\n{}",
            report_path, report_text
        ),
        980,
    )?;

    set_focus_state(
        paths,
        "recovery",
        Some(task.id),
        &task.title,
        "Pending hard reset report loaded into recovery mode.",
    )?;
    let _ = sync_agent_thread(
        paths,
        "recovering",
        "recovery",
        None,
        Some(task.id),
        &format!(
            "Recovery task {} created from pending hard reset report.",
            task.id
        ),
    );
    let _ = record_agent_event(
        paths,
        "loop/recoveryActivated",
        Some(task.id),
        &task.title,
        &summary,
        &serde_json::to_string(&serde_json::json!({
            "reportPath": report_path,
            "taskId": task.id,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );

    let archived_name = format!(
        "handled-hard-reset-report-{}.json",
        now_iso().replace(':', "-").replace('+', "_")
    );
    let archived_path = paths.recovery_dir.join(archived_name);
    let _ = fs::rename(&paths.pending_hard_reset_report_path, archived_path);

    Ok(Some(task))
}

fn sanitize_unclean_runtime(paths: &Paths) -> anyhow::Result<bool> {
    let conn = open_db(paths)?;
    let active_turn_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM agent_turns WHERE status = 'in_progress'",
        [],
        |row| row.get(0),
    )?;
    let active_task_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM tasks WHERE status = 'active'",
        [],
        |row| row.get(0),
    )?;
    if active_turn_count == 0 && active_task_count == 0 {
        return Ok(false);
    }

    if !paths.pending_hard_reset_report_path.exists() {
        let reason = format!(
            "unclean restart detected: {} in-progress turns and {} active tasks remained from the previous process",
            active_turn_count, active_task_count
        );
        let _ = write_pending_hard_reset_report(paths, "startup_unclean_shutdown", &reason);
    }

    let timestamp = now_iso();
    conn.execute(
        "UPDATE agent_turns
         SET updated_at = ?1,
             mode_at_end = 'recovery',
             status = 'interrupted_by_restart',
             summary = COALESCE(summary, 'Turn interrupted by hard restart.'),
             output = COALESCE(output, 'Process restarted before bounded turn completion.'),
             completed_at = ?1
         WHERE status = 'in_progress'",
        params![timestamp],
    )?;
    conn.execute(
        "UPDATE tasks
         SET status = 'queued',
             updated_at = ?1,
             last_checkpoint_summary = COALESCE(last_checkpoint_summary, 'Task requeued after unclean restart.'),
             last_checkpoint_at = COALESCE(last_checkpoint_at, ?1)
         WHERE status = 'active'",
        params![timestamp],
    )?;
    let _ = record_agent_event(
        paths,
        "loop/uncleanRestartSanitized",
        None,
        "",
        "Unclean restart detected; active turns were closed and active tasks were requeued for recovery-aware continuation.",
        &serde_json::to_string(&serde_json::json!({
            "activeTurnCount": active_turn_count,
            "activeTaskCount": active_task_count,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    let _ = register_loop_incident(
        paths,
        "unclean_restart",
        "critical",
        "Unclean restart of the Infinity Loop detected.",
        &format!(
            "The new process found {} in-progress turns and {} active tasks from the previous runtime. Recovery must inspect the pending hard-reset report before normal work resumes.",
            active_turn_count, active_task_count
        ),
        None,
        None,
        true,
        false,
    );
    touch_loop_failure_state(
        paths,
        true,
        "Unclean restart sanitized; recovery required before normal reprioritization.",
        false,
    )?;
    Ok(true)
}

pub fn register_loop_incident(
    paths: &Paths,
    incident_key: &str,
    severity: &str,
    summary: &str,
    detail: &str,
    related_task_id: Option<i64>,
    related_turn_id: Option<i64>,
    hard_reset_required: bool,
    enqueue_self_preservation_task: bool,
) -> anyhow::Result<LoopIncidentRecord> {
    let conn = open_db(paths)?;
    let existing = conn
        .query_row(
            "SELECT id, created_at, updated_at, incident_key, severity, status, summary, detail,
                    related_task_id, related_turn_id, self_preservation_task_id, hard_reset_required,
                    hard_reset_report_path, resolved_at
             FROM loop_incidents
             WHERE incident_key = ?1 AND status = 'open'
             ORDER BY id DESC
             LIMIT 1",
            params![incident_key],
            map_loop_incident_row,
        )
        .optional()?;

    let now = now_iso();
    let incident = if let Some(existing) = existing {
        conn.execute(
            "UPDATE loop_incidents
             SET updated_at = ?2,
                 severity = ?3,
                 summary = ?4,
                 detail = ?5,
                 related_task_id = COALESCE(?6, related_task_id),
                 related_turn_id = COALESCE(?7, related_turn_id),
                 hard_reset_required = CASE WHEN ?8 != 0 THEN 1 ELSE hard_reset_required END
             WHERE id = ?1",
            params![
                existing.id,
                now,
                severity,
                summary,
                detail,
                related_task_id,
                related_turn_id,
                bool_to_i64(hard_reset_required),
            ],
        )?;
        load_loop_incident_by_id(paths, existing.id)?
            .ok_or_else(|| anyhow::anyhow!("failed to reload loop incident {}", existing.id))?
    } else {
        conn.execute(
            "INSERT INTO loop_incidents(
                created_at, updated_at, incident_key, severity, status, summary, detail,
                related_task_id, related_turn_id, hard_reset_required
             ) VALUES(?1, ?2, ?3, ?4, 'open', ?5, ?6, ?7, ?8, ?9)",
            params![
                now,
                now,
                incident_key,
                severity,
                summary,
                detail,
                related_task_id,
                related_turn_id,
                bool_to_i64(hard_reset_required),
            ],
        )?;
        let incident_id = conn.last_insert_rowid();
        let _ = record_agent_event(
            paths,
            "loop/incidentOpened",
            related_task_id,
            "",
            summary,
            &serde_json::to_string(&serde_json::json!({
                "incidentKey": incident_key,
                "severity": severity,
                "hardResetRequired": hard_reset_required,
                "relatedTurnId": related_turn_id,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );
        load_loop_incident_by_id(paths, incident_id)?
            .ok_or_else(|| anyhow::anyhow!("failed to reload loop incident {}", incident_id))?
    };

    let _ = touch_loop_failure_state(
        paths,
        hard_reset_required,
        &format!(
            "Open loop incident {}: {}",
            incident.incident_key, incident.summary
        ),
        false,
    );

    if enqueue_self_preservation_task && incident.self_preservation_task_id.is_none() {
        let task = enqueue_internal_task(
            paths,
            related_task_id,
            "self_preservation",
            format!(
                "Loop self-preservation for incident {}",
                incident.incident_key
            )
            .as_str(),
            &format!(
                "Investigate the open loop incident '{}' with severity {} and stabilize the Infinity Loop in a bounded way. Incident summary: {}\n\n{}",
                incident.incident_key, incident.severity, incident.summary, incident.detail
            ),
            920,
        )?;
        conn.execute(
            "UPDATE loop_incidents
             SET updated_at = ?2,
                 self_preservation_task_id = ?3
             WHERE id = ?1",
            params![incident.id, now_iso(), task.id],
        )?;
        return load_loop_incident_by_id(paths, incident.id)?
            .ok_or_else(|| anyhow::anyhow!("failed to reload loop incident {}", incident.id));
    }

    Ok(incident)
}

pub fn resolve_loop_incident(
    paths: &Paths,
    incident_key: &str,
    resolution_note: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE loop_incidents
         SET updated_at = ?2,
             status = 'resolved',
             resolved_at = ?2,
             detail = detail || '\n\nResolved: ' || ?3
         WHERE incident_key = ?1
           AND status = 'open'",
        params![incident_key, now_iso(), resolution_note],
    )?;
    let _ = record_agent_event(
        paths,
        "loop/incidentResolved",
        None,
        "",
        resolution_note,
        &serde_json::to_string(&serde_json::json!({
            "incidentKey": incident_key,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    Ok(())
}

fn resolve_loop_incidents_for_task(
    paths: &Paths,
    task_id: i64,
    resolution_note: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE loop_incidents
         SET updated_at = ?2,
             status = 'resolved',
             resolved_at = ?2,
             detail = detail || '\n\nResolved: ' || ?3
         WHERE status = 'open'
           AND (self_preservation_task_id = ?1 OR related_task_id = ?1)",
        params![task_id, now_iso(), resolution_note],
    )?;
    Ok(())
}

fn touch_loop_failure_state(
    paths: &Paths,
    hard_reset: bool,
    notes: &str,
    successful_recovery: bool,
) -> anyhow::Result<()> {
    let mut state = load_self_preservation_state(paths);
    let now = now_iso();
    state.last_loop_failure_at = Some(now.clone());
    if hard_reset {
        state.last_hard_reset_at = Some(now.clone());
    }
    if successful_recovery {
        state.successful_self_recoveries += 1;
    }
    state.notes = notes.to_string();
    state.updated_at = now;
    write_self_preservation_state(paths, &state)
}

pub fn start_agent_turn(
    paths: &Paths,
    task_id: i64,
    task_title: &str,
    trigger: &str,
    mode_at_start: &str,
) -> anyhow::Result<AgentTurnRecord> {
    let conn = open_db(paths)?;
    let created_at = now_iso();
    conn.execute(
        "INSERT INTO agent_turns(
            created_at, updated_at, task_id, task_title, trigger, mode_at_start, status
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, 'in_progress')",
        params![
            created_at,
            created_at,
            task_id,
            task_title,
            trigger,
            mode_at_start
        ],
    )?;
    let turn_id = conn.last_insert_rowid();
    let _ = record_agent_event(
        paths,
        "turn/started",
        Some(task_id),
        task_title,
        &format!("Turn {} gestartet.", turn_id),
        &serde_json::to_string(&serde_json::json!({
            "turnId": turn_id,
            "trigger": trigger,
            "modeAtStart": mode_at_start,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    load_agent_turn_by_id(paths, turn_id)?
        .ok_or_else(|| anyhow::anyhow!("failed to reload agent turn {turn_id}"))
}

pub fn complete_agent_turn(
    paths: &Paths,
    turn_id: i64,
    status: &str,
    mode_at_end: &str,
    summary: &str,
    output: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    let completed_at = now_iso();
    conn.execute(
        "UPDATE agent_turns
         SET updated_at = ?2,
             mode_at_end = ?3,
             status = ?4,
             summary = ?5,
             output = ?6,
             completed_at = ?7
         WHERE id = ?1",
        params![
            turn_id,
            completed_at,
            mode_at_end,
            status,
            summary,
            output,
            completed_at
        ],
    )?;
    if let Some(turn) = load_agent_turn_by_id(paths, turn_id)? {
        let _ = record_agent_event(
            paths,
            "turn/completed",
            Some(turn.task_id),
            &turn.task_title,
            &format!("Turn {} completed with status {}.", turn_id, status),
            &serde_json::to_string(&serde_json::json!({
                "turnId": turn_id,
                "status": status,
                "modeAtEnd": mode_at_end,
                "summary": summary,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );
    }
    Ok(())
}

pub fn recover_orphaned_active_turns(
    paths: &Paths,
    live_turn_id: Option<i64>,
    stale_after_secs: u64,
) -> anyhow::Result<Option<String>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, task_id, task_title, trigger, mode_at_start,
                mode_at_end, status, summary, output, completed_at
         FROM agent_turns
         WHERE status = 'in_progress'
         ORDER BY id ASC",
    )?;
    let rows = stmt.query_map([], map_agent_turn_row)?;
    let active_turns: Vec<AgentTurnRecord> = rows.filter_map(Result::ok).collect();
    drop(stmt);

    let mut recovered_turn_ids = Vec::new();
    let mut recovery_notes = Vec::new();

    for turn in active_turns {
        if Some(turn.id) == live_turn_id {
            continue;
        }

        let age_secs = seconds_since_iso(&turn.created_at).unwrap_or_default();
        let summary = if live_turn_id.is_none() {
            format!(
                "Recovered orphaned turn {} for task {} because no live bounded handle exists.",
                turn.id, turn.task_id
            )
        } else {
            format!(
                "Recovered stray turn {} for task {} while live turn {:?} is current.",
                turn.id, turn.task_id, live_turn_id
            )
        };
        let detail = format!(
            "{} The persisted turn has been in progress for {}s and crossed back into reprioritize/recovery.",
            summary, age_secs
        );
        let timestamp = now_iso();

        conn.execute(
            "UPDATE agent_turns
             SET updated_at = ?2,
                 mode_at_end = 'recovery',
                 status = 'interrupted_by_watchdog',
                 summary = COALESCE(summary, ?3),
                 output = COALESCE(output, ?4),
                 completed_at = ?2
             WHERE id = ?1
               AND status = 'in_progress'",
            params![turn.id, timestamp, summary, detail],
        )?;
        conn.execute(
            "UPDATE tasks
             SET status = 'queued',
                 updated_at = ?2,
                 last_checkpoint_summary = COALESCE(last_checkpoint_summary, ?3),
                 last_checkpoint_at = COALESCE(last_checkpoint_at, ?2)
             WHERE id = ?1
               AND status = 'active'",
            params![
                turn.task_id,
                now_iso(),
                format!("Task requeued after orphaned turn {} recovery.", turn.id)
            ],
        )?;
        recovered_turn_ids.push(turn.id);
        recovery_notes.push(format!("#{}:{}:{}s", turn.id, turn.task_id, age_secs));
        let _ = record_agent_event(
            paths,
            "turn/recoveredOrphaned",
            Some(turn.task_id),
            &turn.task_title,
            &summary,
            &serde_json::to_string(&serde_json::json!({
                "turnId": turn.id,
                "taskId": turn.task_id,
                "ageSeconds": age_secs,
                "staleAfterSeconds": stale_after_secs,
                "liveTurnId": live_turn_id,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );
        let _ = register_loop_incident(
            paths,
            "orphaned_active_turn",
            "critical",
            "A persisted bounded turn had no matching live executor and was recovered.",
            &detail,
            Some(turn.task_id),
            Some(turn.id),
            true,
            false,
        );
    }

    if recovered_turn_ids.is_empty() {
        return Ok(None);
    }

    let summary = format!(
        "Recovered orphaned bounded turns and requeued their tasks: {}",
        recovery_notes.join(" | ")
    );
    set_focus_state(paths, "reprioritize", None, "", &summary)?;
    let _ = sync_agent_thread(
        paths,
        "running",
        "reprioritize",
        live_turn_id,
        None,
        &summary,
    );
    Ok(Some(summary))
}

pub fn watchdog_interrupt_live_turn(
    paths: &Paths,
    turn_id: i64,
    task_id: i64,
    task_title: &str,
    age_secs: u64,
    stale_after_secs: u64,
) -> anyhow::Result<Option<String>> {
    let conn = open_db(paths)?;
    let status = conn
        .query_row(
            "SELECT status FROM agent_turns WHERE id = ?1",
            params![turn_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if !matches!(status.as_deref(), Some("in_progress")) {
        return Ok(None);
    }

    let summary = format!(
        "Watchdog interrupted live turn {} for task {} after {}s.",
        turn_id, task_id, age_secs
    );
    let detail = format!(
        "{} The bounded run exceeded the stall threshold of {}s and was actively ejected back into recovery/reprioritize.",
        summary, stale_after_secs
    );
    let timestamp = now_iso();
    conn.execute(
        "UPDATE agent_turns
         SET updated_at = ?2,
             mode_at_end = 'recovery',
             status = 'interrupted_by_watchdog',
             summary = COALESCE(summary, ?3),
             output = COALESCE(output, ?4),
             completed_at = ?2
         WHERE id = ?1
           AND status = 'in_progress'",
        params![turn_id, timestamp, summary, detail],
    )?;

    record_task_checkpoint(paths, task_id, "watchdog_recovery", &summary, &detail)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'queued',
             updated_at = ?2
         WHERE id = ?1
           AND status = 'active'",
        params![task_id, now_iso()],
    )?;

    let recovery_task = enqueue_internal_task(
        paths,
        Some(task_id),
        "recovery",
        "Run watchdog recovery for stale bounded work",
        &format!(
            "The running bounded turn for task #{} {} was aborted by the watchdog after {}s as stale.\n\n{}\n\nAnalyze why the turn got stuck, whether context, tool, browser, model, or runtime caused it, and then decide on a safe re-entry path.",
            task_id, task_title, age_secs, detail
        ),
        980,
    )?;
    set_focus_state(
        paths,
        "recovery",
        Some(recovery_task.id),
        &recovery_task.title,
        &summary,
    )?;
    let _ = sync_agent_thread(
        paths,
        "recovering",
        "recovery",
        None,
        Some(recovery_task.id),
        &summary,
    );
    let _ = record_agent_event(
        paths,
        "turn/watchdogInterrupted",
        Some(task_id),
        task_title,
        &summary,
        &serde_json::to_string(&serde_json::json!({
            "turnId": turn_id,
            "taskId": task_id,
            "ageSeconds": age_secs,
            "staleAfterSeconds": stale_after_secs,
            "recoveryTaskId": recovery_task.id,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    let _ = register_loop_incident(
        paths,
        "active_turn_stall",
        "critical",
        "A live bounded turn exceeded the watchdog threshold and was interrupted.",
        &detail,
        Some(task_id),
        Some(turn_id),
        true,
        false,
    );
    Ok(Some(summary))
}

pub fn interrupt_live_turn_for_signal_preemption(
    paths: &Paths,
    turn_id: i64,
    task_id: i64,
    task_title: &str,
    age_secs: u64,
    queued_task_id: i64,
    queued_task_title: &str,
    signal: &TurnSignalRecord,
) -> anyhow::Result<Option<String>> {
    let conn = open_db(paths)?;
    let status = conn
        .query_row(
            "SELECT status FROM agent_turns WHERE id = ?1",
            params![turn_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if !matches!(status.as_deref(), Some("in_progress")) {
        return Ok(None);
    }

    let message_excerpt = signal.message.trim().chars().take(180).collect::<String>();
    let summary = format!(
        "Interrupt signal {} preempted live turn {} for task {} in favor of queued task {}.",
        signal.id, turn_id, task_id, queued_task_id
    );
    let detail = format!(
        "{} The running turn was active for {}s. Speaker: {} via {}. Queued successor: #{} {}. Signal: {}",
        summary,
        age_secs,
        signal.speaker.trim(),
        signal.source_channel,
        queued_task_id,
        queued_task_title,
        message_excerpt
    );
    let timestamp = now_iso();
    conn.execute(
        "UPDATE agent_turns
         SET updated_at = ?2,
             mode_at_end = 'reprioritize',
             status = 'interrupted_by_signal',
             summary = COALESCE(summary, ?3),
             output = COALESCE(output, ?4),
             completed_at = ?2
         WHERE id = ?1
           AND status = 'in_progress'",
        params![turn_id, timestamp, summary, detail],
    )?;

    record_task_checkpoint(paths, task_id, "signal_preemption", &summary, &detail)?;
    conn.execute(
        "UPDATE tasks
         SET status = 'queued',
             updated_at = ?2
         WHERE id = ?1
           AND status = 'active'",
        params![task_id, now_iso()],
    )?;
    conn.execute(
        "UPDATE turn_signals
         SET status = 'consumed',
             consumed_at = ?2,
             resolution_note = ?3
         WHERE id = ?1
           AND status = 'recorded'",
        params![signal.id, now_iso(), summary],
    )?;

    set_focus_state(
        paths,
        "reprioritize",
        Some(queued_task_id),
        queued_task_title,
        &summary,
    )?;
    let _ = sync_agent_thread(
        paths,
        "running",
        "reprioritize",
        None,
        Some(queued_task_id),
        &summary,
    );
    let _ = record_agent_event(
        paths,
        "turn/preemptedByInterrupt",
        Some(task_id),
        task_title,
        &summary,
        &serde_json::to_string(&serde_json::json!({
            "turnId": turn_id,
            "taskId": task_id,
            "queuedTaskId": queued_task_id,
            "queuedTaskTitle": queued_task_title,
            "signalId": signal.id,
            "signalKind": signal.signal_kind,
            "speaker": signal.speaker,
            "sourceChannel": signal.source_channel,
            "ageSeconds": age_secs,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    Ok(Some(summary))
}

pub fn is_agent_turn_in_progress(paths: &Paths, turn_id: i64) -> anyhow::Result<bool> {
    let conn = open_db(paths)?;
    let status = conn
        .query_row(
            "SELECT status FROM agent_turns WHERE id = ?1",
            params![turn_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    Ok(matches!(status.as_deref(), Some("in_progress")))
}

pub fn list_recent_agent_turns(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<AgentTurnRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, task_id, task_title, trigger, mode_at_start,
                mode_at_end, status, summary, output, completed_at
         FROM agent_turns
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_agent_turn_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn load_active_agent_turn(paths: &Paths) -> anyhow::Result<Option<AgentTurnRecord>> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT id, created_at, updated_at, task_id, task_title, trigger, mode_at_start,
                mode_at_end, status, summary, output, completed_at
         FROM agent_turns
         WHERE status = 'in_progress'
         ORDER BY id DESC
         LIMIT 1",
        [],
        map_agent_turn_row,
    )
    .optional()
    .map_err(Into::into)
}

pub fn load_latest_completed_agent_turn(paths: &Paths) -> anyhow::Result<Option<AgentTurnRecord>> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT id, created_at, updated_at, task_id, task_title, trigger, mode_at_start,
                mode_at_end, status, summary, output, completed_at
         FROM agent_turns
         WHERE status != 'in_progress'
         ORDER BY id DESC
         LIMIT 1",
        [],
        map_agent_turn_row,
    )
    .optional()
    .map_err(Into::into)
}

pub fn load_agent_thread(paths: &Paths) -> anyhow::Result<AgentThreadRecord> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT thread_key, created_at, updated_at, lifecycle_status, current_mode,
                active_turn_id, active_task_id, queue_depth, note
         FROM agent_threads
         WHERE singleton = 1",
        [],
        |row| {
            Ok(AgentThreadRecord {
                thread_key: row.get(0)?,
                created_at: row.get(1)?,
                updated_at: row.get(2)?,
                lifecycle_status: row.get(3)?,
                current_mode: row.get(4)?,
                active_turn_id: row.get(5)?,
                active_task_id: row.get(6)?,
                queue_depth: row.get(7)?,
                note: row.get(8)?,
            })
        },
    )
    .map_err(Into::into)
}

pub fn sync_agent_thread(
    paths: &Paths,
    lifecycle_status: &str,
    current_mode: &str,
    active_turn_id: Option<i64>,
    active_task_id: Option<i64>,
    note: &str,
) -> anyhow::Result<AgentThreadRecord> {
    let conn = open_db(paths)?;
    let queue_depth: i64 = conn.query_row(
        "SELECT COUNT(*) FROM tasks WHERE status IN ('queued', 'active')",
        [],
        |row| row.get(0),
    )?;
    conn.execute(
        "UPDATE agent_threads
         SET updated_at = ?1,
             lifecycle_status = ?2,
             current_mode = ?3,
             active_turn_id = ?4,
             active_task_id = ?5,
             queue_depth = ?6,
             note = ?7
         WHERE singleton = 1",
        params![
            now_iso(),
            lifecycle_status,
            current_mode,
            active_turn_id,
            active_task_id,
            queue_depth,
            note
        ],
    )?;
    load_agent_thread(paths)
}

pub fn record_turn_signal_for_active_turn(
    paths: &Paths,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> anyhow::Result<Option<TurnSignalRecord>> {
    let active_turn = load_active_agent_turn(paths)?;
    let focus = load_focus_state(paths).ok();
    let task_id = active_turn
        .as_ref()
        .map(|turn| turn.task_id)
        .or_else(|| focus.as_ref().and_then(|state| state.active_task_id));
    let task_title = active_turn
        .as_ref()
        .map(|turn| turn.task_title.clone())
        .or_else(|| {
            focus
                .as_ref()
                .map(|state| state.active_task_title.clone())
                .filter(|title| !title.trim().is_empty())
        })
        .unwrap_or_default();

    if task_id.is_none() && active_turn.is_none() {
        return Ok(None);
    }

    let bios = load_bios(paths);
    let signal_kind = classify_turn_signal_kind(&bios.owner.name, source_channel, speaker, message);
    let conn = open_db(paths)?;
    let created_at = now_iso();
    conn.execute(
        "INSERT INTO turn_signals(
            created_at, thread_key, turn_id, task_id, signal_kind, source_channel, speaker, message, status
         ) VALUES(?1, 'main', ?2, ?3, ?4, ?5, ?6, ?7, 'recorded')",
        params![
            created_at,
            active_turn.as_ref().map(|turn| turn.id),
            task_id,
            signal_kind,
            source_channel,
            speaker.trim(),
            message.trim(),
        ],
    )?;
    let signal_id = conn.last_insert_rowid();
    let signal = load_turn_signal_by_id(paths, signal_id)?
        .ok_or_else(|| anyhow::anyhow!("failed to reload turn signal {signal_id}"))?;
    let event_method = if signal.signal_kind == "interrupt" {
        "turn/interrupt"
    } else {
        "turn/steer"
    };
    let _ = record_agent_event(
        paths,
        event_method,
        signal.task_id,
        &task_title,
        &format!(
            "The running thread received a {} signal from {} via {}.",
            signal.signal_kind,
            speaker.trim(),
            source_channel
        ),
        &serde_json::to_string(&serde_json::json!({
            "signalId": signal.id,
            "turnId": signal.turn_id,
            "taskId": signal.task_id,
            "signalKind": signal.signal_kind,
            "speaker": speaker.trim(),
            "sourceChannel": source_channel,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    Ok(Some(signal))
}

pub fn list_recent_turn_signals(
    paths: &Paths,
    limit: usize,
) -> anyhow::Result<Vec<TurnSignalRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, thread_key, turn_id, task_id, signal_kind, source_channel,
                speaker, message, status, consumed_at, resolution_note
         FROM turn_signals
         ORDER BY id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_turn_signal_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn list_turn_signals_for_task(
    paths: &Paths,
    task_id: i64,
    limit: usize,
) -> anyhow::Result<Vec<TurnSignalRecord>> {
    let conn = open_db(paths)?;
    let mut stmt = conn.prepare(
        "SELECT id, created_at, thread_key, turn_id, task_id, signal_kind, source_channel,
                speaker, message, status, consumed_at, resolution_note
         FROM turn_signals
         WHERE task_id = ?1
         ORDER BY id DESC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![task_id, limit as i64], map_turn_signal_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

pub fn latest_context_package_for_task(
    paths: &Paths,
    task_id: i64,
) -> anyhow::Result<Option<ContextPackageRecord>> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT id, created_at, task_id, task_title, context_mode, budget_hint, rationale, package_json
         FROM context_packages
         WHERE task_id = ?1
         ORDER BY id DESC
         LIMIT 1",
        params![task_id],
        |row| {
            Ok(ContextPackageRecord {
                id: row.get(0)?,
                created_at: row.get(1)?,
                task_id: row.get(2)?,
                task_title: row.get(3)?,
                context_mode: row.get(4)?,
                budget_hint: row.get(5)?,
                rationale: row.get(6)?,
                package_json: row.get(7)?,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

fn open_db(paths: &Paths) -> anyhow::Result<Connection> {
    if let Some(parent) = paths.runtime_db_path.parent() {
        fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(&paths.runtime_db_path)?;
    conn.busy_timeout(Duration::from_secs(5))?;
    conn.pragma_update(None, "foreign_keys", "ON")?;
    conn.pragma_update(None, "temp_store", "MEMORY")?;
    Ok(conn)
}

fn insert_bootstrap_task(conn: &Connection, task: &BootstrapTaskTemplate) -> anyhow::Result<i64> {
    let created_at = now_iso();
    conn.execute(
        "INSERT INTO tasks(
            created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
            title, detail, trust_level, priority_score, status
         ) VALUES(?1, ?2, NULL, NULL, NULL, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'queued')",
        params![
            created_at,
            created_at,
            task.source_channel,
            task.speaker,
            task.task_kind,
            task.title,
            task.detail,
            task.trust_level,
            task.priority_score,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub(crate) fn load_task_by_id(paths: &Paths, task_id: i64) -> anyhow::Result<Option<TaskRecord>> {
    let conn = open_db(paths)?;
    load_task_by_id_with_conn(&conn, task_id)
}

fn load_task_by_id_with_conn(
    conn: &Connection,
    task_id: i64,
) -> anyhow::Result<Option<TaskRecord>> {
    conn.query_row(
        "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count,
                last_checkpoint_summary, last_checkpoint_at, last_output
         FROM tasks
         WHERE id = ?1",
        params![task_id],
        map_task_row,
    )
    .optional()
    .map_err(Into::into)
}

fn map_task_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TaskRecord> {
    Ok(TaskRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        parent_task_id: row.get(3)?,
        worker_job_id: row.get(4)?,
        source_interrupt_id: row.get(5)?,
        source_channel: row.get(6)?,
        speaker: row.get(7)?,
        task_kind: row.get(8)?,
        title: row.get(9)?,
        detail: row.get(10)?,
        trust_level: row.get(11)?,
        priority_score: row.get(12)?,
        status: row.get(13)?,
        run_count: row.get(14)?,
        last_checkpoint_summary: row.get(15)?,
        last_checkpoint_at: row.get(16)?,
        last_output: row.get(17)?,
    })
}

fn map_worker_job_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<WorkerJobRecord> {
    Ok(WorkerJobRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        parent_task_id: row.get(3)?,
        parent_task_title: row.get(4)?,
        worker_kind: row.get(5)?,
        contract_title: row.get(6)?,
        contract_detail: row.get(7)?,
        status: row.get(8)?,
        request_note: row.get(9)?,
        result_summary: row.get(10)?,
        result_detail: row.get(11)?,
        review_summary: row.get(12)?,
        review_detail: row.get(13)?,
        review_task_id: row.get(14)?,
        completed_at: row.get(15)?,
    })
}

fn map_agent_event_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentEventRecord> {
    Ok(AgentEventRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        method: row.get(2)?,
        active_task_id: row.get(3)?,
        active_task_title: row.get(4)?,
        body: row.get(5)?,
        payload_json: row.get(6)?,
    })
}

fn map_agent_turn_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<AgentTurnRecord> {
    Ok(AgentTurnRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        task_id: row.get(3)?,
        task_title: row.get(4)?,
        trigger: row.get(5)?,
        mode_at_start: row.get(6)?,
        mode_at_end: row.get(7)?,
        status: row.get(8)?,
        summary: row.get(9)?,
        output: row.get(10)?,
        completed_at: row.get(11)?,
    })
}

fn map_turn_signal_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<TurnSignalRecord> {
    Ok(TurnSignalRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        thread_key: row.get(2)?,
        turn_id: row.get(3)?,
        task_id: row.get(4)?,
        signal_kind: row.get(5)?,
        source_channel: row.get(6)?,
        speaker: row.get(7)?,
        message: row.get(8)?,
        status: row.get(9)?,
        consumed_at: row.get(10)?,
        resolution_note: row.get(11)?,
    })
}

fn map_loop_incident_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<LoopIncidentRecord> {
    Ok(LoopIncidentRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        incident_key: row.get(3)?,
        severity: row.get(4)?,
        status: row.get(5)?,
        summary: row.get(6)?,
        detail: row.get(7)?,
        related_task_id: row.get(8)?,
        related_turn_id: row.get(9)?,
        self_preservation_task_id: row.get(10)?,
        hard_reset_required: row.get::<_, i64>(11)? != 0,
        hard_reset_report_path: row.get(12)?,
        resolved_at: row.get(13)?,
    })
}

fn set_focus_state(
    paths: &Paths,
    mode: &str,
    active_task_id: Option<i64>,
    active_task_title: &str,
    note: &str,
) -> anyhow::Result<()> {
    let conn = open_db(paths)?;
    let previous_mode: String = conn.query_row(
        "SELECT mode FROM focus_state WHERE singleton = 1",
        [],
        |row| row.get(0),
    )?;
    let queue_depth: i64 = conn.query_row(
        "SELECT COUNT(*) FROM tasks WHERE status IN ('queued', 'active')",
        [],
        |row| row.get(0),
    )?;
    if previous_mode != mode {
        conn.execute(
            "INSERT INTO mode_transitions(created_at, from_mode, to_mode, active_task_id, active_task_title, trigger, note)
             VALUES(?1, ?2, ?3, ?4, ?5, 'system_mode_change', ?6)",
            params![now_iso(), previous_mode, mode, active_task_id, active_task_title, note],
        )?;
    }
    conn.execute(
        "UPDATE focus_state
         SET mode = ?1,
             active_task_id = ?2,
             active_task_title = ?3,
             queue_depth = ?4,
             last_reprioritized_at = ?5,
             note = ?6
         WHERE singleton = 1",
        params![
            mode,
            active_task_id,
            active_task_title,
            queue_depth,
            now_iso(),
            note
        ],
    )?;
    if previous_mode != mode
        && should_publish_mode_change_event(
            &previous_mode,
            mode,
            active_task_id,
            active_task_title,
            note,
        )
    {
        let _ = record_agent_event(
            paths,
            "mode/changed",
            active_task_id,
            active_task_title,
            note,
            &serde_json::to_string(&serde_json::json!({
                "fromMode": previous_mode,
                "toMode": mode,
            }))
            .unwrap_or_else(|_| "{}".to_string()),
        );
    }
    Ok(())
}

fn is_internal_maintenance_mode(mode: &str) -> bool {
    matches!(mode, "observe" | "reprioritize")
}

fn is_standard_maintenance_note(note: &str) -> bool {
    matches!(
        note.trim(),
        "Observe current signals, resources and queued work before reprioritization."
            | "Task priorities recalculated."
            | "No queued tasks available."
    )
}

fn should_publish_mode_change_event(
    previous_mode: &str,
    mode: &str,
    active_task_id: Option<i64>,
    active_task_title: &str,
    note: &str,
) -> bool {
    if is_internal_maintenance_mode(mode) && is_standard_maintenance_note(note) {
        return false;
    }
    if !is_internal_maintenance_mode(previous_mode) && !is_internal_maintenance_mode(mode) {
        return true;
    }
    if active_task_id.is_some() || !active_task_title.trim().is_empty() {
        return true;
    }
    let normalized_note = note.trim();
    if normalized_note.is_empty() {
        return false;
    }
    !is_standard_maintenance_note(normalized_note)
}

fn create_task_from_interrupt(
    paths: &Paths,
    interrupt: &LoopInterruptRecord,
) -> anyhow::Result<TaskRecord> {
    let conn = open_db(paths)?;
    create_task_from_interrupt_with_conn(paths, &conn, interrupt)
}

fn create_task_from_interrupt_with_conn(
    paths: &Paths,
    conn: &Connection,
    interrupt: &LoopInterruptRecord,
) -> anyhow::Result<TaskRecord> {
    let bios = load_bios(paths);
    let owner_name = bios.owner.name;
    let task_kind = classify_interrupt_task_kind(
        &owner_name,
        &interrupt.source_channel,
        &interrupt.speaker,
        &interrupt.message,
    );
    let trust_level =
        classify_trust_level(&owner_name, &interrupt.source_channel, &interrupt.speaker);
    let loop_safety = load_loop_safety_policy(paths);
    let priority_score = compute_priority_score(
        &loop_safety,
        &owner_name,
        &task_kind,
        &interrupt.source_channel,
        &interrupt.speaker,
        &interrupt.message,
    );
    let title = build_interrupt_task_title(&interrupt.source_channel, &task_kind, &interrupt.message);
    let created_at = now_iso();
    conn.execute(
        "INSERT INTO tasks(
            created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
            title, detail, trust_level, priority_score, status
         ) VALUES(?1, ?2, NULL, NULL, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, 'queued')",
        params![
            created_at,
            created_at,
            interrupt.id,
            interrupt.source_channel,
            interrupt.speaker,
            task_kind,
            title,
            interrupt.message,
            trust_level,
            priority_score
        ],
    )?;
    let task_id = conn.last_insert_rowid();
    let task = load_task_by_id_with_conn(conn, task_id)?
        .ok_or_else(|| anyhow::anyhow!("failed to reload created task {task_id}"))?;
    let _ = record_agent_event_with_conn(
        conn,
        "task/queued",
        Some(task.id),
        &task.title,
        "Interrupt was converted into a queued task.",
        &serde_json::to_string(&serde_json::json!({
            "sourceChannel": task.source_channel,
            "speaker": task.speaker,
            "taskKind": task.task_kind,
            "priorityScore": task.priority_score,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    Ok(task)
}

pub fn record_agent_event(
    paths: &Paths,
    method: &str,
    active_task_id: Option<i64>,
    active_task_title: &str,
    body: &str,
    payload_json: &str,
) -> anyhow::Result<i64> {
    let conn = open_db(paths)?;
    record_agent_event_with_conn(
        &conn,
        method,
        active_task_id,
        active_task_title,
        body,
        payload_json,
    )
}

fn record_agent_event_with_conn(
    conn: &Connection,
    method: &str,
    active_task_id: Option<i64>,
    active_task_title: &str,
    body: &str,
    payload_json: &str,
) -> anyhow::Result<i64> {
    conn.execute(
        "INSERT INTO agent_events(created_at, method, active_task_id, active_task_title, body, payload_json)
         VALUES(?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            now_iso(),
            method,
            active_task_id,
            active_task_title,
            body,
            payload_json
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

pub fn enqueue_internal_task(
    paths: &Paths,
    parent_task_id: Option<i64>,
    task_kind: &str,
    title: &str,
    detail: &str,
    priority_score: i64,
) -> anyhow::Result<TaskRecord> {
    let conn = open_db(paths)?;
    {
        let existing_id = conn
            .query_row(
                "SELECT id
                 FROM tasks
                 WHERE task_kind = ?1
                   AND source_channel = 'system_guard'
                   AND status IN ('queued', 'active', 'await_review')
                   AND (
                        (parent_task_id IS NULL AND ?2 IS NULL)
                        OR parent_task_id = ?2
                   )
                 ORDER BY priority_score DESC, id DESC
                 LIMIT 1",
                params![task_kind, parent_task_id],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if let Some(task_id) = existing_id {
            return load_task_by_id(paths, task_id)?.ok_or_else(|| {
                anyhow::anyhow!("failed to reload deduped internal task {task_id}")
            });
        }
    }
    {
        let blocked_cutoff = recent_past_iso(blocked_internal_task_reuse_window_secs());
        let existing_id = conn
            .query_row(
                "SELECT id
                 FROM tasks
                 WHERE task_kind = ?1
                   AND source_channel = 'system_guard'
                   AND status = 'blocked'
                   AND updated_at >= ?3
                   AND (
                        (parent_task_id IS NULL AND ?2 IS NULL)
                        OR parent_task_id = ?2
                   )
                 ORDER BY priority_score DESC, id DESC
                 LIMIT 1",
                params![task_kind, parent_task_id, blocked_cutoff],
                |row| row.get::<_, i64>(0),
            )
            .optional()?;
        if let Some(task_id) = existing_id {
            return load_task_by_id(paths, task_id)?.ok_or_else(|| {
                anyhow::anyhow!("failed to reload recently blocked internal task {task_id}")
            });
        }
    }
    let created_at = now_iso();
    conn.execute(
        "INSERT INTO tasks(
            created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
            title, detail, trust_level, priority_score, status
         ) VALUES(?1, ?2, ?3, NULL, NULL, 'system_guard', 'system_guard', ?4, ?5, ?6, 'system', ?7, 'queued')",
        params![
            created_at,
            created_at,
            parent_task_id,
            task_kind,
            title,
            detail,
            priority_score,
        ],
    )?;
    let task_id = conn.last_insert_rowid();
    let task = load_task_by_id(paths, task_id)?
        .ok_or_else(|| anyhow::anyhow!("failed to reload internal task {task_id}"))?;
    let _ = record_agent_event(
        paths,
        "task/internalQueued",
        Some(task.id),
        &task.title,
        "Systemischer Selbsterhaltungs- oder Guard-Task wurde eingereiht.",
        &serde_json::to_string(&serde_json::json!({
            "taskKind": task.task_kind,
            "parentTaskId": parent_task_id,
            "priorityScore": priority_score,
        }))
        .unwrap_or_else(|_| "{}".to_string()),
    );
    Ok(task)
}

pub fn emit_self_review_task(
    paths: &Paths,
    parent_task: &TaskRecord,
    result_summary: &str,
    result_detail: &str,
    output: Option<&str>,
) -> anyhow::Result<TaskRecord> {
    let conn = open_db(paths)?;
    let existing_id = conn
        .query_row(
            "SELECT id
             FROM tasks
             WHERE task_kind = 'self_review'
               AND parent_task_id = ?1
               AND status IN ('queued', 'active', 'await_review')
             ORDER BY id DESC
             LIMIT 1",
            params![parent_task.id],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    if let Some(task_id) = existing_id {
        return load_task_by_id(paths, task_id)?
            .ok_or_else(|| anyhow::anyhow!("failed to reload self-review task {task_id}"));
    }

    record_task_checkpoint(
        paths,
        parent_task.id,
        "await_review",
        result_summary,
        result_detail,
    )?;
    let timestamp = now_iso();
    conn.execute(
        "UPDATE tasks
         SET status = 'await_review',
             updated_at = ?2,
             last_output = ?3
         WHERE id = ?1",
        params![parent_task.id, timestamp, output.unwrap_or("")],
    )?;

    let mut review_detail = format!(
        "Mandatory self-review for task #{task_id} {title}.\n\nPreliminary result:\n{result_summary}\n\nDetail:\n{result_detail}\n\nCheck the real state against the actual task. Return `completionReview` with `decision=approve|revise|blocked`, a concrete `note`, and optional `evidenceGaps` plus `confidence`. The parent task may finish only if you explicitly return `completionReview.decision=approve`. If completion is not reliable yet, reopen the work with `completionReview.decision=revise`.",
        task_id = parent_task.id,
        title = parent_task.title,
        result_summary = result_summary,
        result_detail = result_detail,
    );
    if parent_task.task_kind == "homepage_bridge" {
        review_detail.push_str(
            "\n\nRequirement for this review: use real browser work instead of mere claims. If you have not actually looked at the current homepage or BIOS, first use `browserAction=inspect_visual` on `https://127.0.0.1:8443/` or `https://127.0.0.1:8443/bios`, and optionally provide `browserQuestion` so the visible UI is judged through the Qwen3.5 vision path. If the local vision kleinhirn is not active yet, request `brainAction=upgrade_local_browser_vision_kleinhirn` first. Plain `browserAction` screenshots count only as artifacts here, not as completed visual verification. After that, continue with another review step using `taskStatus=continue`."
        );
    }

    let priority_score = parent_task.priority_score.clamp(890, 980);
    conn.execute(
        "INSERT INTO tasks(
            created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
            title, detail, trust_level, priority_score, status
         ) VALUES(?1, ?2, ?3, NULL, NULL, 'system_guard', 'self_review', 'self_review', ?4, ?5, 'system', ?6, 'queued')",
        params![
            timestamp,
            timestamp,
            parent_task.id,
            format!("Self-review task #{} before completion", parent_task.id),
            review_detail,
            priority_score,
        ],
    )?;
    let task_id = conn.last_insert_rowid();
    load_task_by_id(paths, task_id)?
        .ok_or_else(|| anyhow::anyhow!("failed to reload self-review task {task_id}"))
}

fn load_agent_turn_by_id(paths: &Paths, turn_id: i64) -> anyhow::Result<Option<AgentTurnRecord>> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT id, created_at, updated_at, task_id, task_title, trigger, mode_at_start,
                mode_at_end, status, summary, output, completed_at
         FROM agent_turns
         WHERE id = ?1",
        params![turn_id],
        map_agent_turn_row,
    )
    .optional()
    .map_err(Into::into)
}

fn seconds_since_iso(value: &str) -> Option<i64> {
    let parsed = DateTime::parse_from_rfc3339(value).ok()?;
    Some((Utc::now() - parsed.with_timezone(&Utc)).num_seconds())
}

fn load_loop_incident_by_id(
    paths: &Paths,
    incident_id: i64,
) -> anyhow::Result<Option<LoopIncidentRecord>> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT id, created_at, updated_at, incident_key, severity, status, summary, detail,
                related_task_id, related_turn_id, self_preservation_task_id, hard_reset_required,
                hard_reset_report_path, resolved_at
         FROM loop_incidents
         WHERE id = ?1",
        params![incident_id],
        map_loop_incident_row,
    )
    .optional()
    .map_err(Into::into)
}

fn load_turn_signal_by_id(
    paths: &Paths,
    signal_id: i64,
) -> anyhow::Result<Option<TurnSignalRecord>> {
    let conn = open_db(paths)?;
    conn.query_row(
        "SELECT id, created_at, thread_key, turn_id, task_id, signal_kind, source_channel,
                speaker, message, status, consumed_at, resolution_note
         FROM turn_signals
         WHERE id = ?1",
        params![signal_id],
        map_turn_signal_row,
    )
    .optional()
    .map_err(Into::into)
}

pub(crate) fn claim_next_worker_job(paths: &Paths) -> anyhow::Result<Option<WorkerJobRecord>> {
    let mut conn = open_db(paths)?;
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let job_id = tx
        .query_row(
            "SELECT id
             FROM worker_jobs
             WHERE status = 'queued'
             ORDER BY id ASC
             LIMIT 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    let Some(job_id) = job_id else {
        tx.commit()?;
        return Ok(None);
    };
    let claimed = tx.execute(
        "UPDATE worker_jobs
         SET status = 'running',
             updated_at = ?2
         WHERE id = ?1
           AND status = 'queued'",
        params![job_id, now_iso()],
    )?;
    if claimed == 0 {
        tx.commit()?;
        return Ok(None);
    }
    let job = load_worker_job_by_id_with_conn(&tx, job_id)?;
    tx.commit()?;
    Ok(job)
}

fn load_worker_job_by_id(
    paths: &Paths,
    worker_job_id: i64,
) -> anyhow::Result<Option<WorkerJobRecord>> {
    let conn = open_db(paths)?;
    load_worker_job_by_id_with_conn(&conn, worker_job_id)
}

fn load_worker_job_by_id_with_conn(
    conn: &Connection,
    worker_job_id: i64,
) -> anyhow::Result<Option<WorkerJobRecord>> {
    conn.query_row(
        "SELECT id, created_at, updated_at, parent_task_id, parent_task_title, worker_kind,
                contract_title, contract_detail, status, request_note, result_summary,
                result_detail, review_summary, review_detail, review_task_id, completed_at
         FROM worker_jobs
         WHERE id = ?1",
        params![worker_job_id],
        map_worker_job_row,
    )
    .optional()
    .map_err(Into::into)
}

pub(crate) fn emit_worker_review_task(
    paths: &Paths,
    job: &WorkerJobRecord,
    result_summary: &str,
    result_detail: &str,
) -> anyhow::Result<TaskRecord> {
    let conn = open_db(paths)?;
    let created_at = now_iso();
    let detail = format!(
        "Delegated worker job #{job_id} requires review.\n\nParent Task: #{parent_task_id} {parent_task_title}\nWorker Kind: {worker_kind}\nContract Title: {contract_title}\nContract Detail: {contract_detail}\n\nWorker Summary: {result_summary}\n\nWorker Detail:\n{result_detail}\n\nReturn `completionReview` with `decision=approve|revise|blocked`, a concrete `note`, and optional `evidenceGaps` plus `confidence`. The parent task may finish only if you explicitly return `completionReview.decision=approve`.",
        job_id = job.id,
        parent_task_id = job.parent_task_id,
        parent_task_title = job.parent_task_title,
        worker_kind = job.worker_kind,
        contract_title = job.contract_title,
        contract_detail = job.contract_detail,
        result_summary = result_summary,
        result_detail = result_detail,
    );
    let priority_score = 900_i64;
    conn.execute(
        "INSERT INTO tasks(
            created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
            title, detail, trust_level, priority_score, status
         ) VALUES(?1, ?2, ?3, ?4, NULL, 'worker_review', ?5, 'worker_review', ?6, ?7, 'system', ?8, 'queued')",
        params![
            created_at,
            created_at,
            job.parent_task_id,
            job.id,
            format!("worker::{}", job.worker_kind),
            format!("Review delegated worker job #{}", job.id),
            detail,
            priority_score,
        ],
    )?;
    let task_id = conn.last_insert_rowid();
    load_task_by_id(paths, task_id)?
        .ok_or_else(|| anyhow::anyhow!("failed to reload worker review task {task_id}"))
}

pub(crate) fn finalize_worker_job_for_review(
    paths: &Paths,
    worker_job_id: i64,
    result_summary: &str,
    result_detail: &str,
    review_task_id: i64,
) -> anyhow::Result<WorkerJobRecord> {
    let conn = open_db(paths)?;
    conn.execute(
        "UPDATE worker_jobs
         SET status = 'await_review',
             updated_at = ?2,
             result_summary = ?3,
             result_detail = ?4,
             review_task_id = ?5
         WHERE id = ?1",
        params![
            worker_job_id,
            now_iso(),
            result_summary,
            result_detail,
            review_task_id
        ],
    )?;
    load_worker_job_by_id(paths, worker_job_id)?
        .ok_or_else(|| anyhow::anyhow!("failed to reload worker job {worker_job_id}"))
}

fn bool_to_i64(value: bool) -> i64 {
    if value { 1 } else { 0 }
}

fn classify_task_kind(message: &str) -> String {
    let lowered = message.to_lowercase();
    if contains_any(
        &lowered,
        &[
            "infinity loop",
            "always-on",
            "always on",
            "watchdog",
            "healthz",
            "readyz",
            "heartbeat",
            "selbsterhalt",
            "self-preservation",
            "self preservation",
            "stuck",
            "haengt",
            "hängt",
            "absturz",
            "crash",
            "restart",
            "neustart",
            "health check",
            "healthcheck",
        ],
    ) {
        return "self_preservation".to_string();
    }
    if contains_any(
        &lowered,
        &[
            "grosshirn",
            "großhirn",
            "gpt-5.4",
            "gpt 5.4",
            "openai",
            "api token",
            "api-key",
            "api key",
            "sk-proj-",
        ],
    ) && contains_any(
        &lowered,
        &[
            "wechsel",
            "wechsle",
            "umschalten",
            "schalte",
            "switch",
            "aktivier",
            "aktiviere",
            "konfigurier",
            "konfiguriere",
        ],
    ) {
        return "grosshirn_activation".to_string();
    }
    if crate::brain_runtime::extract_requested_local_kleinhirn_model(message).is_some()
        && contains_any(
            &lowered,
            &[
                "wechsel",
                "wechsle",
                "umschalten",
                "schalte",
                "switch",
                "aktivier",
                "aktiviere",
            ],
        )
    {
        return "local_model_switch".to_string();
    }
    if contains_any(
        &lowered,
        &[
            "execcommand",
            "execsession",
            "command-exec",
            "repo",
            "readme",
            "bericht",
            "report",
        ],
    ) {
        return "owner_interrupt".to_string();
    }
    if contains_any(
        &lowered,
        &[
            "homepage",
            "startseite",
            "webseite",
            "bios",
            "sichtbar",
            "branding",
        ],
    ) {
        return "homepage_bridge".to_string();
    }
    if contains_any(
        &lowered,
        &[
            "superpassword",
            "root-auth",
            "root auth",
            "root-trust",
            "root trust",
            "passwort",
        ],
    ) {
        return "root_trust".to_string();
    }
    if contains_any(
        &lowered,
        &[
            "modell",
            "kleinhirn",
            "grosshirn",
            "großhirn",
            "qwen",
            "gpt-oss",
        ],
    ) {
        return "model_or_resource".to_string();
    }
    if contains_any(
        &lowered,
        &["mail", "email", "whatsapp", "kanal", "kommunikationspfad"],
    ) {
        return "channel_expansion".to_string();
    }
    "owner_interrupt".to_string()
}

fn classify_interrupt_task_kind(
    owner_name: &str,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> String {
    if deterministic_chat_parsing_disabled(source_channel) {
        return "owner_interrupt".to_string();
    }

    let direct_terminal_surface = matches!(source_channel, "terminal" | "attach_terminal");
    let owner_signal = is_owner_message(speaker, owner_name);
    let trusted_owner_surface = matches!(source_channel, "bios" | "homepage") && owner_signal;

    if direct_terminal_surface || trusted_owner_surface || owner_signal {
        let lowered = message.to_lowercase();
        if contains_any(
            &lowered,
            &[
                "grosshirn",
                "großhirn",
                "gpt-5.4",
                "gpt 5.4",
                "openai",
                "api token",
                "api-key",
                "api key",
                "sk-proj-",
            ],
        ) && contains_any(
            &lowered,
            &[
                "wechsel",
                "wechsle",
                "umschalten",
                "schalte",
                "switch",
                "aktivier",
                "aktiviere",
                "konfigurier",
                "konfiguriere",
            ],
        ) {
            return "grosshirn_activation".to_string();
        }
        if crate::brain_runtime::extract_requested_local_kleinhirn_model(message).is_some()
            && contains_any(
                &lowered,
                &[
                    "wechsel",
                    "wechsle",
                    "umschalten",
                    "schalte",
                    "switch",
                    "aktivier",
                    "aktiviere",
                ],
            )
        {
            return "local_model_switch".to_string();
        }
        return "owner_interrupt".to_string();
    }

    classify_task_kind(message)
}

fn classify_trust_level(owner_name: &str, source_channel: &str, speaker: &str) -> String {
    if source_channel == "terminal" || source_channel == "attach_terminal" {
        return "system".to_string();
    }
    if source_channel == "bios" || source_channel == "homepage" {
        return if is_owner_message(speaker, owner_name) {
            "owner_trust".to_string()
        } else {
            "trusted".to_string()
        };
    }
    if is_owner_message(speaker, owner_name) {
        return "owner_external".to_string();
    }
    "low_trust".to_string()
}

fn classify_turn_signal_kind(
    owner_name: &str,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> String {
    if deterministic_chat_parsing_disabled(source_channel) {
        return "interrupt".to_string();
    }

    let lowered = message.to_lowercase();
    let owner_signal = is_owner_message(speaker, owner_name);
    let urgent = contains_any(
        &lowered,
        &[
            "dringend",
            "sofort",
            "kritisch",
            "notfall",
            "incident",
            "prod",
            "produktion",
            "jetzt",
        ],
    );
    let system_level = matches!(source_channel, "terminal" | "attach_terminal");
    if urgent || (owner_signal && system_level) {
        "interrupt".to_string()
    } else {
        "steer".to_string()
    }
}

fn compute_priority_score(
    loop_safety: &LoopSafetyPolicy,
    owner_name: &str,
    task_kind: &str,
    source_channel: &str,
    speaker: &str,
    message: &str,
) -> i64 {
    let mut score = 0;
    let owner_override = is_owner_message(speaker, owner_name);
    let grosshirn_activation = task_kind == "grosshirn_activation";
    if owner_override {
        score += 500;
    }
    score += match source_channel {
        "worker_review" => 340,
        "bios" => 280,
        "homepage" => 240,
        "terminal" | "attach_terminal" => 220,
        "email" => 80,
        "whatsapp" => 60,
        _ => 40,
    };

    if deterministic_chat_parsing_disabled(source_channel) {
        if grosshirn_activation {
            score += 180;
        }
        if is_self_preservation_task_kind(task_kind) {
            score = score.max(loop_safety.self_preservation_priority_floor);
        }
        if owner_override {
            score = score.max(loop_safety.owner_override_priority_floor);
            if grosshirn_activation {
                score = score.max(loop_safety.owner_override_priority_floor + 180);
            }
        }
        return score;
    }

    let lowered = message.to_lowercase();
    if contains_any(
        &lowered,
        &[
            "dringend",
            "sofort",
            "kritisch",
            "ausfall",
            "incident",
            "security",
            "prod",
            "produktion",
        ],
    ) {
        score += 260;
    }
    if contains_any(
        &lowered,
        &["homepage", "bios", "branding", "superpassword", "passwort"],
    ) {
        score += 80;
    }
    if contains_any(
        &lowered,
        &[
            "worker-job",
            "review",
            "delegiert",
            "delegierten worker-job",
        ],
    ) {
        score += 220;
    }
    if grosshirn_activation {
        score += 180;
    }
    if is_self_preservation_task_kind(task_kind) {
        score = score.max(loop_safety.self_preservation_priority_floor);
        if contains_any(
            &lowered,
            &[
                "watchdog",
                "healthz",
                "readyz",
                "heartbeat",
                "crash",
                "absturz",
                "restart",
                "stuck",
                "infinity loop",
            ],
        ) {
            score += 120;
        }
    }
    if owner_override {
        score = score.max(loop_safety.owner_override_priority_floor);
        if grosshirn_activation {
            score = score.max(loop_safety.owner_override_priority_floor + 180);
        }
    }
    score
}

fn build_task_title(task_kind: &str, message: &str) -> String {
    let canned = match task_kind {
        "self_preservation" => Some("Secure Infinity Loop self-preservation"),
        "local_model_switch" => Some("Execute the local kleinhirn model switch"),
        "homepage_bridge" => Some("Revise the homepage or BIOS bridge"),
        "root_trust" => Some("Calibrate root trust and superpassword"),
        "grosshirn_activation" => Some("Activate and configure grosshirn mode"),
        "model_or_resource" => Some("Handle the kleinhirn or resource question"),
        "channel_expansion" => Some("Assess communication paths"),
        _ => None,
    };
    if let Some(title) = canned {
        return title.to_string();
    }
    summarize_for_memory(message)
}

fn build_interrupt_task_title(source_channel: &str, task_kind: &str, message: &str) -> String {
    if source_channel == "attach_terminal" {
        return "Chatten".to_string();
    }
    if source_channel == "bios" {
        return "BIOS-Chat".to_string();
    }
    build_task_title(task_kind, message)
}

fn deterministic_chat_parsing_disabled(source_channel: &str) -> bool {
    matches!(source_channel, "attach_terminal" | "bios")
}

fn is_self_preservation_task_kind(task_kind: &str) -> bool {
    matches!(
        task_kind,
        "self_preservation" | "bootstrap_runtime_guard" | "bootstrap_supervisor"
    )
}

fn contains_any(text: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| text.contains(needle))
}

fn compute_owner_commitment_score(
    owner_name: &str,
    owner_contact_established: bool,
    bios_primary_channel_confirmed: bool,
    superpassword_set: bool,
) -> i64 {
    let mut score = 0;
    if !owner_name.trim().is_empty() {
        score += 20;
    }
    if owner_contact_established {
        score += 30;
    }
    if bios_primary_channel_confirmed {
        score += 25;
    }
    if superpassword_set {
        score += 25;
    }
    score
}

fn normalize_identity(value: &str) -> String {
    value
        .chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

fn is_owner_message(speaker: &str, owner_name: &str) -> bool {
    let speaker_norm = normalize_identity(speaker);
    let owner_norm = normalize_identity(owner_name);
    let canonical_owner_norm = normalize_identity("Michael Welsch");
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

fn summarize_for_memory(message: &str) -> String {
    let trimmed = message.trim();
    if trimmed.chars().count() <= 140 {
        trimmed.to_string()
    } else {
        trimmed.chars().take(140).collect::<String>() + "..."
    }
}

fn normalize_learning_class(value: &str) -> String {
    match value.trim().to_lowercase().as_str() {
        "operational" | "ops" | "essential" | "daily_ops" | "daily_operations" => {
            "operational".to_string()
        }
        "negative" | "anti" | "anti_pattern" | "antipattern" | "failure" | "failed" => {
            "negative".to_string()
        }
        _ => "general".to_string(),
    }
}

fn list_active_learning_entries_with_conn(
    conn: &Connection,
    limit: usize,
) -> anyhow::Result<Vec<LearningEntryRecord>> {
    let mut stmt = conn.prepare(
        "SELECT id, created_at, updated_at, source_task_id, source_turn_id, source_task_kind,
                source_channel, learning_class, status, summary, detail, evidence, applicability,
                confidence, salience, recall_count, last_recalled_at, source
         FROM learning_entries
         WHERE status = 'active'
         ORDER BY CASE learning_class
                    WHEN 'operational' THEN 0
                    WHEN 'negative' THEN 1
                    ELSE 2
                  END,
                  salience DESC,
                  confidence DESC,
                  updated_at DESC,
                  id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_learning_entry_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

fn load_learning_entry_by_id_with_conn(
    conn: &Connection,
    entry_id: i64,
) -> anyhow::Result<Option<LearningEntryRecord>> {
    conn.query_row(
        "SELECT id, created_at, updated_at, source_task_id, source_turn_id, source_task_kind,
                source_channel, learning_class, status, summary, detail, evidence, applicability,
                confidence, salience, recall_count, last_recalled_at, source
         FROM learning_entries
         WHERE id = ?1",
        params![entry_id],
        map_learning_entry_row,
    )
    .optional()
    .map_err(Into::into)
}

fn map_learning_entry_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<LearningEntryRecord> {
    Ok(LearningEntryRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        source_task_id: row.get(3)?,
        source_turn_id: row.get(4)?,
        source_task_kind: row.get(5)?,
        source_channel: row.get(6)?,
        learning_class: row.get(7)?,
        status: row.get(8)?,
        summary: row.get(9)?,
        detail: row.get(10)?,
        evidence: row.get(11)?,
        applicability: row.get(12)?,
        confidence: row.get(13)?,
        salience: row.get(14)?,
        recall_count: row.get(15)?,
        last_recalled_at: row.get(16)?,
        source: row.get(17)?,
    })
}

fn upsert_memory_summary_with_conn(
    conn: &Connection,
    scope: &str,
    summary: &str,
) -> anyhow::Result<()> {
    conn.execute(
        "INSERT INTO memory_summaries(scope, updated_at, summary)
         VALUES(?1, ?2, ?3)
         ON CONFLICT(scope) DO UPDATE SET
             updated_at = excluded.updated_at,
             summary = excluded.summary",
        params![scope, now_iso(), summary],
    )?;
    Ok(())
}

fn refresh_learning_memory_summaries_with_conn(conn: &Connection) -> anyhow::Result<()> {
    let active = list_active_learning_entries_with_conn(conn, 24)?;
    let summarize_class = |class_name: &str, fallback: &str| {
        let summaries = active
            .iter()
            .filter(|entry| entry.learning_class == class_name)
            .take(5)
            .map(|entry| entry.summary.clone())
            .collect::<Vec<_>>();
        if summaries.is_empty() {
            fallback.to_string()
        } else {
            summaries.join(" | ")
        }
    };
    let working_set = if active.is_empty() {
        "No permanently activated learning is recorded in the learning path yet.".to_string()
    } else {
        active
            .iter()
            .take(6)
            .map(|entry| {
                let prefix = match entry.learning_class.as_str() {
                    "operational" => "[ops]",
                    "negative" => "[neg]",
                    _ => "[gen]",
                };
                format!("{prefix} {}", entry.summary)
            })
            .collect::<Vec<_>>()
            .join(" | ")
    };
    upsert_memory_summary_with_conn(conn, "learning_working_set", &working_set)?;
    upsert_memory_summary_with_conn(
        conn,
        "learning_operational",
        &summarize_class(
            "operational",
            "No operational core learnings exist in the daily working set yet.",
        ),
    )?;
    upsert_memory_summary_with_conn(
        conn,
        "learning_general",
        &summarize_class(
            "general",
            "No general learnings exist in the learning path yet.",
        ),
    )?;
    upsert_memory_summary_with_conn(
        conn,
        "learning_negative",
        &summarize_class(
            "negative",
            "No negative learnings or anti-patterns exist in the learning path yet.",
        ),
    )?;
    Ok(())
}

fn map_person_profile_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<PersonProfileRecord> {
    Ok(PersonProfileRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        canonical_key: row.get(3)?,
        display_name: row.get(4)?,
        primary_email: row.get(5)?,
        relationship_kind: row.get(6)?,
        trust_level: row.get(7)?,
        last_interaction_at: row.get(8)?,
        last_channel: row.get(9)?,
        interaction_count: row.get(10)?,
        conversation_memory_summary: row.get(11)?,
        notebook_summary: row.get(12)?,
        proactive_guard_note: row.get(13)?,
    })
}

fn map_person_note_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<PersonNoteRecord> {
    Ok(PersonNoteRecord {
        id: row.get(0)?,
        person_profile_id: row.get(1)?,
        person_display_name: row.get(2)?,
        created_at: row.get(3)?,
        updated_at: row.get(4)?,
        note_kind: row.get(5)?,
        source_channel: row.get(6)?,
        source_ref: row.get(7)?,
        summary: row.get(8)?,
        detail: row.get(9)?,
        important: row.get::<_, i64>(10)? != 0,
    })
}

fn map_proactive_contact_candidate_row(
    row: &rusqlite::Row<'_>,
) -> rusqlite::Result<ProactiveContactCandidateRecord> {
    Ok(ProactiveContactCandidateRecord {
        id: row.get(0)?,
        created_at: row.get(1)?,
        updated_at: row.get(2)?,
        person_profile_id: row.get(3)?,
        person_display_name: row.get(4)?,
        person_email: row.get(5)?,
        source_task_id: row.get(6)?,
        source_turn_id: row.get(7)?,
        status: row.get(8)?,
        channel: row.get(9)?,
        subject: row.get(10)?,
        draft_body: row.get(11)?,
        rationale: row.get(12)?,
        conflict_check: row.get(13)?,
        requires_grosshirn_validation: row.get::<_, i64>(14)? != 0,
        validation_task_id: row.get(15)?,
        validated_at: row.get(16)?,
        validation_decision: row.get(17)?,
        validation_note: row.get(18)?,
        dispatch_task_id: row.get(19)?,
        dispatched_at: row.get(20)?,
        dispatch_channel: row.get(21)?,
        dispatch_note: row.get(22)?,
        outbound_message_id: row.get(23)?,
        source: row.get(24)?,
    })
}

fn list_recent_person_notes_with_conn(
    conn: &Connection,
    limit: usize,
) -> anyhow::Result<Vec<PersonNoteRecord>> {
    let mut stmt = conn.prepare(
        "SELECT n.id, n.person_profile_id, p.display_name, n.created_at, n.updated_at,
                n.note_kind, n.source_channel, n.source_ref, n.summary, n.detail, n.important
         FROM person_notes n
         JOIN person_profiles p ON p.id = n.person_profile_id
         ORDER BY n.updated_at DESC, n.id DESC
         LIMIT ?1",
    )?;
    let rows = stmt.query_map(params![limit as i64], map_person_note_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

fn list_person_notes_for_person_with_conn(
    conn: &Connection,
    person_profile_id: i64,
    limit: usize,
) -> anyhow::Result<Vec<PersonNoteRecord>> {
    let mut stmt = conn.prepare(
        "SELECT n.id, n.person_profile_id, p.display_name, n.created_at, n.updated_at,
                n.note_kind, n.source_channel, n.source_ref, n.summary, n.detail, n.important
         FROM person_notes n
         JOIN person_profiles p ON p.id = n.person_profile_id
         WHERE n.person_profile_id = ?1
         ORDER BY n.updated_at DESC, n.id DESC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(
        params![person_profile_id, limit as i64],
        map_person_note_row,
    )?;
    Ok(rows.filter_map(Result::ok).collect())
}

fn list_proactive_contact_candidates_with_conn(
    conn: &Connection,
    limit: usize,
    pending_only: bool,
) -> anyhow::Result<Vec<ProactiveContactCandidateRecord>> {
    let sql = if pending_only {
        "SELECT id, created_at, updated_at, person_profile_id, person_display_name, person_email,
                source_task_id, source_turn_id, status, channel, subject, draft_body, rationale,
                conflict_check, requires_grosshirn_validation, validation_task_id, validated_at,
                validation_decision, validation_note, dispatch_task_id, dispatched_at,
                dispatch_channel, dispatch_note, outbound_message_id, source
         FROM proactive_contact_candidates
         WHERE status IN ('pending_validation', 'needs_revision', 'approved')
         ORDER BY CASE status
                    WHEN 'pending_validation' THEN 0
                    WHEN 'needs_revision' THEN 1
                    WHEN 'approved' THEN 2
                    ELSE 3
                  END,
                  updated_at DESC,
                  id DESC
         LIMIT ?1"
    } else {
        "SELECT id, created_at, updated_at, person_profile_id, person_display_name, person_email,
                source_task_id, source_turn_id, status, channel, subject, draft_body, rationale,
                conflict_check, requires_grosshirn_validation, validation_task_id, validated_at,
                validation_decision, validation_note, dispatch_task_id, dispatched_at,
                dispatch_channel, dispatch_note, outbound_message_id, source
         FROM proactive_contact_candidates
         ORDER BY CASE status
                    WHEN 'pending_validation' THEN 0
                    WHEN 'needs_revision' THEN 1
                    WHEN 'approved' THEN 2
                    WHEN 'sent' THEN 3
                    WHEN 'send_failed' THEN 4
                    WHEN 'dispatch_blocked' THEN 5
                    WHEN 'rejected' THEN 6
                    ELSE 7
                  END,
                  updated_at DESC,
                  id DESC
         LIMIT ?1"
    };
    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![limit as i64], map_proactive_contact_candidate_row)?;
    Ok(rows.filter_map(Result::ok).collect())
}

fn load_person_profile_by_id_with_conn(
    conn: &Connection,
    person_profile_id: i64,
) -> anyhow::Result<Option<PersonProfileRecord>> {
    conn.query_row(
        "SELECT id, created_at, updated_at, canonical_key, display_name, primary_email,
                relationship_kind, trust_level, last_interaction_at, last_channel,
                interaction_count, conversation_memory_summary, notebook_summary,
                proactive_guard_note
         FROM person_profiles
         WHERE id = ?1",
        params![person_profile_id],
        map_person_profile_row,
    )
    .optional()
    .map_err(Into::into)
}

fn load_person_profile_by_key_with_conn(
    conn: &Connection,
    canonical_key: &str,
) -> anyhow::Result<Option<PersonProfileRecord>> {
    conn.query_row(
        "SELECT id, created_at, updated_at, canonical_key, display_name, primary_email,
                relationship_kind, trust_level, last_interaction_at, last_channel,
                interaction_count, conversation_memory_summary, notebook_summary,
                proactive_guard_note
         FROM person_profiles
         WHERE canonical_key = ?1",
        params![canonical_key],
        map_person_profile_row,
    )
    .optional()
    .map_err(Into::into)
}

fn load_proactive_contact_candidate_by_id_with_conn(
    conn: &Connection,
    candidate_id: i64,
) -> anyhow::Result<Option<ProactiveContactCandidateRecord>> {
    conn.query_row(
        "SELECT id, created_at, updated_at, person_profile_id, person_display_name, person_email,
                source_task_id, source_turn_id, status, channel, subject, draft_body, rationale,
                conflict_check, requires_grosshirn_validation, validation_task_id, validated_at,
                validation_decision, validation_note, dispatch_task_id, dispatched_at,
                dispatch_channel, dispatch_note, outbound_message_id, source
         FROM proactive_contact_candidates
         WHERE id = ?1",
        params![candidate_id],
        map_proactive_contact_candidate_row,
    )
    .optional()
    .map_err(Into::into)
}

fn load_proactive_contact_candidate_by_validation_task_with_conn(
    conn: &Connection,
    validation_task_id: i64,
) -> anyhow::Result<Option<ProactiveContactCandidateRecord>> {
    conn.query_row(
        "SELECT id, created_at, updated_at, person_profile_id, person_display_name, person_email,
                source_task_id, source_turn_id, status, channel, subject, draft_body, rationale,
                conflict_check, requires_grosshirn_validation, validation_task_id, validated_at,
                validation_decision, validation_note, dispatch_task_id, dispatched_at,
                dispatch_channel, dispatch_note, outbound_message_id, source
         FROM proactive_contact_candidates
         WHERE validation_task_id = ?1
         ORDER BY updated_at DESC, id DESC
         LIMIT 1",
        params![validation_task_id],
        map_proactive_contact_candidate_row,
    )
    .optional()
    .map_err(Into::into)
}

fn load_proactive_contact_candidate_by_dispatch_task_with_conn(
    conn: &Connection,
    dispatch_task_id: i64,
) -> anyhow::Result<Option<ProactiveContactCandidateRecord>> {
    conn.query_row(
        "SELECT id, created_at, updated_at, person_profile_id, person_display_name, person_email,
                source_task_id, source_turn_id, status, channel, subject, draft_body, rationale,
                conflict_check, requires_grosshirn_validation, validation_task_id, validated_at,
                validation_decision, validation_note, dispatch_task_id, dispatched_at,
                dispatch_channel, dispatch_note, outbound_message_id, source
         FROM proactive_contact_candidates
         WHERE dispatch_task_id = ?1
         ORDER BY updated_at DESC, id DESC
         LIMIT 1",
        params![dispatch_task_id],
        map_proactive_contact_candidate_row,
    )
    .optional()
    .map_err(Into::into)
}

#[derive(Debug, Clone)]
struct ResolvedPersonIdentity {
    canonical_key: String,
    display_name: String,
    primary_email: String,
    relationship_kind: String,
    trust_level: String,
}

fn record_person_interaction_with_conn(
    conn: &Connection,
    paths: &Paths,
    source_channel: &str,
    speaker: &str,
    message: &str,
    source_ref: &str,
) -> anyhow::Result<Option<PersonProfileRecord>> {
    let Some(identity) = resolve_person_identity(paths, speaker) else {
        return Ok(None);
    };
    let Some(profile) =
        upsert_person_profile_with_conn(conn, &identity, source_channel, Some(now_iso()))?
    else {
        return Ok(None);
    };
    record_person_note_with_conn(
        conn,
        profile.id,
        "conversation",
        source_channel,
        source_ref,
        &summarize_for_memory(message),
        message.trim(),
        profile.relationship_kind == "owner" || matches!(source_channel, "bios" | "homepage"),
    )?;
    refresh_person_profile_summaries_with_conn(conn, profile.id)?;
    refresh_people_memory_summary_with_conn(conn)?;
    load_person_profile_by_id_with_conn(conn, profile.id)
}

fn store_person_learning_refs_with_conn(
    paths: &Paths,
    conn: &Connection,
    task: &TaskRecord,
    entries: &[LearningEntryRecord],
) -> anyhow::Result<usize> {
    let Some(identity) = resolve_person_identity(paths, &task.speaker) else {
        return Ok(0);
    };
    let Some(profile) =
        upsert_person_profile_with_conn(conn, &identity, &task.source_channel, None)?
    else {
        return Ok(0);
    };
    let mut inserted = 0usize;
    for entry in entries.iter().filter(|entry| entry.status == "active") {
        let detail = format!(
            "Learning-Klasse: {class}\nConfidence: {confidence:.2}\nApplicability: {applicability}\nEvidence: {evidence}\nSource task: #{task_id} {task_title}\n\n{detail}",
            class = entry.learning_class,
            confidence = entry.confidence,
            applicability = entry.applicability,
            evidence = entry.evidence,
            task_id = task.id,
            task_title = task.title,
            detail = entry.detail,
        );
        record_person_note_with_conn(
            conn,
            profile.id,
            "learning_ref",
            &task.source_channel,
            &format!("learning:{}", entry.id),
            &entry.summary,
            &detail,
            entry.learning_class == "operational" || entry.learning_class == "negative",
        )?;
        inserted += 1;
    }
    if inserted > 0 {
        refresh_person_profile_summaries_with_conn(conn, profile.id)?;
        refresh_people_memory_summary_with_conn(conn)?;
    }
    Ok(inserted)
}

fn upsert_person_profile_with_conn(
    conn: &Connection,
    identity: &ResolvedPersonIdentity,
    source_channel: &str,
    last_interaction_at: Option<String>,
) -> anyhow::Result<Option<PersonProfileRecord>> {
    if identity.canonical_key.trim().is_empty() {
        return Ok(None);
    }
    let timestamp = now_iso();
    let initial_interaction_count = if last_interaction_at.is_some() { 1 } else { 0 };
    conn.execute(
        "INSERT INTO person_profiles(
            created_at, updated_at, canonical_key, display_name, primary_email, relationship_kind,
            trust_level, last_interaction_at, last_channel, interaction_count,
            conversation_memory_summary, notebook_summary, proactive_guard_note
         ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, '', '', 'Proactive contact only after grosshirn validation and conflict-of-interest review.')
         ON CONFLICT(canonical_key) DO UPDATE SET
             updated_at = excluded.updated_at,
             display_name = CASE
                 WHEN person_profiles.display_name = '' THEN excluded.display_name
                 ELSE person_profiles.display_name
             END,
             primary_email = CASE
                 WHEN person_profiles.primary_email = '' THEN excluded.primary_email
                 ELSE person_profiles.primary_email
             END,
             relationship_kind = CASE
                 WHEN person_profiles.relationship_kind = 'unknown' THEN excluded.relationship_kind
                 ELSE person_profiles.relationship_kind
             END,
             trust_level = CASE
                 WHEN person_profiles.trust_level IN ('low', 'unknown') AND excluded.trust_level NOT IN ('', 'low', 'unknown')
                     THEN excluded.trust_level
                 ELSE person_profiles.trust_level
             END,
             last_interaction_at = COALESCE(excluded.last_interaction_at, person_profiles.last_interaction_at),
             last_channel = CASE
                 WHEN excluded.last_channel = '' THEN person_profiles.last_channel
                 ELSE excluded.last_channel
             END,
             interaction_count = CASE
                 WHEN excluded.last_interaction_at IS NULL THEN person_profiles.interaction_count
                 ELSE person_profiles.interaction_count + 1
             END,
             proactive_guard_note = CASE
                 WHEN person_profiles.proactive_guard_note = '' THEN excluded.proactive_guard_note
                 ELSE person_profiles.proactive_guard_note
             END",
        params![
            timestamp,
            timestamp,
            identity.canonical_key,
            identity.display_name,
            identity.primary_email,
            identity.relationship_kind,
            identity.trust_level,
            last_interaction_at,
            source_channel,
            initial_interaction_count,
        ],
    )?;
    load_person_profile_by_key_with_conn(conn, &identity.canonical_key)
}

fn record_person_note_with_conn(
    conn: &Connection,
    person_profile_id: i64,
    note_kind: &str,
    source_channel: &str,
    source_ref: &str,
    summary: &str,
    detail: &str,
    important: bool,
) -> anyhow::Result<()> {
    let timestamp = now_iso();
    let existing_id = if source_ref.trim().is_empty() {
        None
    } else {
        conn.query_row(
            "SELECT id
             FROM person_notes
             WHERE person_profile_id = ?1
               AND note_kind = ?2
               AND source_ref = ?3
             ORDER BY id DESC
             LIMIT 1",
            params![person_profile_id, note_kind, source_ref.trim()],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
    };
    if let Some(note_id) = existing_id {
        conn.execute(
            "UPDATE person_notes
             SET updated_at = ?2,
                 source_channel = ?3,
                 summary = ?4,
                 detail = ?5,
                 important = ?6
             WHERE id = ?1",
            params![
                note_id,
                timestamp,
                source_channel,
                summary.trim(),
                detail.trim(),
                bool_to_i64(important),
            ],
        )?;
    } else {
        conn.execute(
            "INSERT INTO person_notes(
                person_profile_id, created_at, updated_at, note_kind, source_channel,
                source_ref, summary, detail, important
             ) VALUES(?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
            params![
                person_profile_id,
                timestamp,
                timestamp,
                note_kind,
                source_channel,
                source_ref.trim(),
                summary.trim(),
                detail.trim(),
                bool_to_i64(important),
            ],
        )?;
    }
    Ok(())
}

fn refresh_person_profile_summaries_with_conn(
    conn: &Connection,
    person_profile_id: i64,
) -> anyhow::Result<()> {
    let conversation = summarize_person_note_block(
        conn,
        person_profile_id,
        &["conversation"],
        "No conversation trail exists for this person yet.",
    )?;
    let notebook = summarize_person_note_block(
        conn,
        person_profile_id,
        &["learning_ref", "notebook"],
        "No learning or notebook references exist for this person yet.",
    )?;
    conn.execute(
        "UPDATE person_profiles
         SET updated_at = ?2,
             conversation_memory_summary = ?3,
             notebook_summary = ?4
         WHERE id = ?1",
        params![person_profile_id, now_iso(), conversation, notebook],
    )?;
    Ok(())
}

fn summarize_person_note_block(
    conn: &Connection,
    person_profile_id: i64,
    note_kinds: &[&str],
    fallback: &str,
) -> anyhow::Result<String> {
    let mut parts = Vec::new();
    for note_kind in note_kinds {
        let mut stmt = conn.prepare(
            "SELECT summary
             FROM person_notes
             WHERE person_profile_id = ?1
               AND note_kind = ?2
             ORDER BY important DESC, updated_at DESC, id DESC
             LIMIT 3",
        )?;
        let rows = stmt.query_map(params![person_profile_id, note_kind], |row| {
            row.get::<_, String>(0)
        })?;
        for summary in rows.filter_map(Result::ok) {
            if !summary.trim().is_empty() && !parts.iter().any(|entry: &String| entry == &summary) {
                parts.push(summary);
            }
            if parts.len() >= 4 {
                break;
            }
        }
        if parts.len() >= 4 {
            break;
        }
    }
    if parts.is_empty() {
        Ok(fallback.to_string())
    } else {
        Ok(parts.join(" | "))
    }
}

fn refresh_people_memory_summary_with_conn(conn: &Connection) -> anyhow::Result<()> {
    let mut stmt = conn.prepare(
        "SELECT display_name, conversation_memory_summary, notebook_summary, relationship_kind, interaction_count
         FROM person_profiles
         ORDER BY CASE relationship_kind
                    WHEN 'owner' THEN 0
                    WHEN 'reports_to' THEN 1
                    WHEN 'ceo' THEN 2
                    WHEN 'board' THEN 3
                    ELSE 4
                  END,
                  interaction_count DESC,
                  updated_at DESC,
                  id DESC
         LIMIT 5",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, i64>(4)?,
        ))
    })?;
    let mut fragments = Vec::new();
    for (display_name, conversation_summary, notebook_summary, _relationship_kind, _count) in
        rows.filter_map(Result::ok)
    {
        if display_name.trim().is_empty() {
            continue;
        }
        let note = if is_placeholder_person_summary(&notebook_summary) {
            conversation_summary
        } else {
            notebook_summary
        };
        fragments.push(format!("{display_name}: {note}"));
    }
    let summary = if fragments.is_empty() {
        "No people working set recorded yet.".to_string()
    } else {
        fragments.join(" | ")
    };
    upsert_memory_summary_with_conn(conn, "people_working_set", &summary)?;
    Ok(())
}

fn resolve_person_identity(paths: &Paths, raw_identity: &str) -> Option<ResolvedPersonIdentity> {
    let trimmed = raw_identity.trim();
    if trimmed.is_empty() {
        return None;
    }
    let normalized_speaker = normalize_identity(trimmed);
    if normalized_speaker.is_empty()
        || matches!(
            normalized_speaker.as_str(),
            "ctoagent" | "systemguard" | "system" | "selfreview" | "workerreview"
        )
    {
        return None;
    }
    let bios = load_bios(paths);
    let organigram = load_organigram(paths);
    let detected_email = extract_email_address(trimmed);
    let mut display_name = extract_person_display_name(trimmed, detected_email.as_deref());
    let mut primary_email = detected_email.unwrap_or_default();
    let mut relationship_kind = "unknown".to_string();
    let mut trust_level = "low".to_string();

    let owner_name = first_non_empty(&[organigram.owner.name.trim(), bios.owner.name.trim()]);
    let owner_email = first_non_empty(&[organigram.owner.email.trim(), bios.owner.email.trim()]);
    if is_identity_match(trimmed, owner_name)
        || (!primary_email.is_empty() && emails_match(&primary_email, owner_email))
        || is_owner_message(trimmed, owner_name)
    {
        relationship_kind = "owner".to_string();
        trust_level = "owner".to_string();
        if display_name.is_empty() {
            display_name = owner_name.to_string();
        }
        if primary_email.is_empty() {
            primary_email = normalize_email_address(owner_email);
        }
    } else if name_matches_any(trimmed, &[organigram.reports_to.as_str()]) {
        relationship_kind = "reports_to".to_string();
        trust_level = "high".to_string();
        if display_name.is_empty() {
            display_name = organigram.reports_to.trim().to_string();
        }
    } else if name_matches_any(trimmed, &[organigram.ceo.as_str()]) {
        relationship_kind = "ceo".to_string();
        trust_level = "high".to_string();
        if display_name.is_empty() {
            display_name = organigram.ceo.trim().to_string();
        }
    } else if let Some(matched) = find_matching_name(trimmed, &organigram.board) {
        relationship_kind = "board".to_string();
        trust_level = "high".to_string();
        if display_name.is_empty() {
            display_name = matched;
        }
    } else if let Some(matched) = find_matching_name(trimmed, &organigram.peer_cxos) {
        relationship_kind = "peer_cxo".to_string();
        trust_level = "medium".to_string();
        if display_name.is_empty() {
            display_name = matched;
        }
    } else if let Some(matched) = find_matching_name(trimmed, &organigram.subordinates.people) {
        relationship_kind = "subordinate_person".to_string();
        trust_level = "medium".to_string();
        if display_name.is_empty() {
            display_name = matched;
        }
    } else if let Some(matched) = find_matching_name(trimmed, &organigram.subordinates.vendors) {
        relationship_kind = "vendor".to_string();
        trust_level = "low".to_string();
        if display_name.is_empty() {
            display_name = matched;
        }
    }

    if display_name.is_empty() {
        display_name = if !primary_email.is_empty() {
            primary_email
                .split('@')
                .next()
                .unwrap_or(trimmed)
                .replace('.', " ")
        } else {
            trimmed.to_string()
        };
    }
    let canonical_key = if !primary_email.is_empty() {
        format!("email:{}", normalize_email_address(&primary_email))
    } else {
        format!("name:{}", normalize_identity(&display_name))
    };
    Some(ResolvedPersonIdentity {
        canonical_key,
        display_name,
        primary_email: normalize_email_address(&primary_email),
        relationship_kind,
        trust_level,
    })
}

fn first_non_empty<'a>(values: &[&'a str]) -> &'a str {
    values
        .iter()
        .copied()
        .find(|value| !value.trim().is_empty())
        .unwrap_or("")
}

fn extract_email_address(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        return None;
    }
    for token in trimmed.split(|c: char| {
        c.is_whitespace() || matches!(c, '<' | '>' | '(' | ')' | ',' | ';' | '"' | '\'')
    }) {
        let normalized = normalize_email_address(token);
        if normalized.contains('@') && normalized.contains('.') {
            return Some(normalized);
        }
    }
    None
}

fn extract_person_display_name(value: &str, email: Option<&str>) -> String {
    let trimmed = value.trim();
    if let Some((name_part, _rest)) = trimmed.split_once('<') {
        let candidate = name_part.trim().trim_matches('"');
        if !candidate.is_empty() {
            return candidate.to_string();
        }
    }
    if let Some(email) = email {
        if trimmed.eq_ignore_ascii_case(email) {
            return email
                .split('@')
                .next()
                .unwrap_or(trimmed)
                .replace('.', " ")
                .trim()
                .to_string();
        }
    }
    trimmed.to_string()
}

fn normalize_email_address(value: &str) -> String {
    value
        .trim()
        .trim_matches(|c: char| matches!(c, '<' | '>' | '"' | '\'' | ',' | ';'))
        .to_lowercase()
}

fn emails_match(left: &str, right: &str) -> bool {
    let left = normalize_email_address(left);
    let right = normalize_email_address(right);
    !left.is_empty() && !right.is_empty() && left == right
}

fn is_identity_match(left: &str, right: &str) -> bool {
    !right.trim().is_empty() && normalize_identity(left) == normalize_identity(right)
}

fn name_matches_any(value: &str, candidates: &[&str]) -> bool {
    candidates
        .iter()
        .any(|candidate| !candidate.trim().is_empty() && is_identity_match(value, candidate))
}

fn find_matching_name(value: &str, candidates: &[String]) -> Option<String> {
    candidates
        .iter()
        .find(|candidate| is_identity_match(value, candidate))
        .cloned()
}

fn normalize_proactive_contact_channel(channel: &str, person_email: &str) -> String {
    let normalized = channel.trim().to_lowercase();
    match normalized.as_str() {
        "email" | "mail" => "email".to_string(),
        "bios" => "bios".to_string(),
        "homepage" => "homepage".to_string(),
        "terminal" => "terminal".to_string(),
        "chat" => "homepage".to_string(),
        _ if !person_email.trim().is_empty() => "email".to_string(),
        _ => "bios".to_string(),
    }
}

fn normalize_proactive_validation_decision(value: &str) -> String {
    match value.trim().to_lowercase().as_str() {
        "approve" | "approved" | "send" | "allow" => "approve".to_string(),
        "reject" | "rejected" | "deny" | "drop" => "reject".to_string(),
        _ => "revise".to_string(),
    }
}

fn normalize_proactive_dispatch_status(value: &str) -> String {
    match value.trim().to_lowercase().as_str() {
        "sent" | "ok" | "delivered" => "sent".to_string(),
        "blocked" | "dispatch_blocked" => "dispatch_blocked".to_string(),
        _ => "send_failed".to_string(),
    }
}

fn is_placeholder_person_summary(value: &str) -> bool {
    value.trim().is_empty() || value.starts_with("No ") || value.starts_with("Noch keine ")
}

fn table_exists(conn: &Connection, table_name: &str) -> anyhow::Result<bool> {
    let exists = conn
        .query_row(
            "SELECT 1
             FROM sqlite_master
             WHERE type = 'table'
               AND name = ?1
             LIMIT 1",
            params![table_name],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    Ok(exists)
}

fn column_exists(conn: &Connection, table_name: &str, column_name: &str) -> anyhow::Result<bool> {
    let mut stmt = conn.prepare(&format!("PRAGMA table_info({table_name})"))?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(1))?;
    Ok(rows.filter_map(Result::ok).any(|name| name == column_name))
}

fn add_column_if_missing(
    conn: &Connection,
    table_name: &str,
    column_name: &str,
    alter_sql: &str,
) -> anyhow::Result<()> {
    if column_exists(conn, table_name, column_name)? {
        return Ok(());
    }
    conn.execute(alter_sql, [])?;
    Ok(())
}

fn claim_loop_interrupt_with_conn(
    conn: &Connection,
    interrupt_id: i64,
) -> anyhow::Result<Option<LoopInterruptRecord>> {
    let claimed = conn.execute(
        "UPDATE loop_interrupts
         SET status = 'processing',
             claimed_at = ?2
         WHERE id = ?1
           AND status = 'pending'",
        params![interrupt_id, now_iso()],
    )?;
    if claimed == 0 {
        return Ok(None);
    }
    load_loop_interrupt_by_id_with_conn(conn, interrupt_id)
}

fn claim_next_loop_interrupt_with_conn(
    conn: &Connection,
) -> anyhow::Result<Option<LoopInterruptRecord>> {
    let next_id = conn
        .query_row(
            "SELECT id
             FROM loop_interrupts
             WHERE status = 'pending'
             ORDER BY id ASC
             LIMIT 1",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()?;
    match next_id {
        Some(interrupt_id) => claim_loop_interrupt_with_conn(conn, interrupt_id),
        None => Ok(None),
    }
}

fn refresh_owner_memory_summary_with_conn(conn: &Connection) -> anyhow::Result<()> {
    let mut stmt = conn.prepare(
        "SELECT summary
         FROM memory_items
         WHERE kind = 'owner_calibration'
         ORDER BY id DESC
         LIMIT 5",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut summaries: Vec<String> = rows.filter_map(Result::ok).collect();
    summaries.reverse();
    let summary = if summaries.is_empty() {
        "No calibration summary recorded yet.".to_string()
    } else {
        summaries.join(" | ")
    };
    let now = now_iso();
    conn.execute(
        "INSERT INTO memory_summaries(scope, updated_at, summary)
         VALUES('owner_calibration', ?1, ?2)
         ON CONFLICT(scope) DO UPDATE SET
             updated_at = excluded.updated_at,
             summary = excluded.summary",
        params![now, summary.clone()],
    )?;
    conn.execute(
        "UPDATE owner_trust
         SET calibration_notes = ?1
         WHERE singleton = 1",
        params![summary],
    )?;
    Ok(())
}

fn record_progress_journal_with_conn(
    conn: &Connection,
    summary: &str,
    detail: &str,
    source: &str,
) -> anyhow::Result<()> {
    conn.execute(
        "INSERT INTO memory_items(created_at, kind, summary, detail, source, important)
         VALUES(?1, 'progress_journal', ?2, ?3, ?4, 1)",
        params![now_iso(), summary, detail, source],
    )?;
    refresh_progress_memory_summary_with_conn(conn)?;
    Ok(())
}

fn refresh_progress_memory_summary_with_conn(conn: &Connection) -> anyhow::Result<()> {
    let mut stmt = conn.prepare(
        "SELECT summary
         FROM memory_items
         WHERE kind = 'progress_journal'
         ORDER BY id DESC
         LIMIT 8",
    )?;
    let rows = stmt.query_map([], |row| row.get::<_, String>(0))?;
    let mut summaries: Vec<String> = rows.filter_map(Result::ok).collect();
    summaries.reverse();
    let summary = if summaries.is_empty() {
        "No improvement journal recorded yet.".to_string()
    } else {
        summaries.join(" | ")
    };
    conn.execute(
        "INSERT INTO memory_summaries(scope, updated_at, summary)
         VALUES('self_improvement', ?1, ?2)
         ON CONFLICT(scope) DO UPDATE SET
             updated_at = excluded.updated_at,
             summary = excluded.summary",
        params![now_iso(), summary],
    )?;
    Ok(())
}

fn upsert_resource(
    conn: &Connection,
    category: &str,
    name: &str,
    observed_at: &str,
    status: &str,
    detail: &str,
) -> anyhow::Result<()> {
    conn.execute(
        "INSERT INTO resources(category, name, observed_at, status, detail)
         VALUES(?1, ?2, ?3, ?4, ?5)
         ON CONFLICT(category, name) DO UPDATE SET
             observed_at = excluded.observed_at,
             status = excluded.status,
             detail = excluded.detail",
        params![category, name, observed_at, status, detail],
    )?;
    Ok(())
}

fn load_owner_trust_row(conn: &Connection) -> anyhow::Result<Option<OwnerTrustSnapshot>> {
    let row = conn
        .query_row(
            "SELECT owner_name,
                    committed_owner_name,
                    owner_contact_established,
                    bios_primary_channel_confirmed,
                    superpassword_set,
                    owner_commitment_score,
                    last_owner_dialogue_at,
                    calibration_notes,
                    brain_access_mode
             FROM owner_trust
             WHERE singleton = 1",
            [],
            |row| {
                Ok(OwnerTrustSnapshot {
                    owner_name: row.get(0)?,
                    committed_owner_name: row.get(1)?,
                    owner_contact_established: row.get::<_, i64>(2)? != 0,
                    bios_primary_channel_confirmed: row.get::<_, i64>(3)? != 0,
                    superpassword_set: row.get::<_, i64>(4)? != 0,
                    owner_commitment_score: row.get(5)?,
                    last_owner_dialogue_at: row.get(6)?,
                    calibration_notes: row.get(7)?,
                    brain_access_mode: row.get(8)?,
                })
            },
        )
        .optional()?;
    Ok(row)
}

impl Default for OwnerTrustSnapshot {
    fn default() -> Self {
        Self {
            owner_name: String::new(),
            committed_owner_name: String::new(),
            owner_contact_established: false,
            bios_primary_channel_confirmed: false,
            superpassword_set: false,
            owner_commitment_score: 0,
            last_owner_dialogue_at: None,
            calibration_notes: String::new(),
            brain_access_mode: "kleinhirn_only".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::Paths;
    use crate::contracts::default_loop_safety_policy;
    use crate::contracts::ensure_contract_files;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::time::SystemTime;
    use std::time::UNIX_EPOCH;

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn unique_test_root(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!(
            "cto_agent_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
    }

    struct EnvGuard(Option<std::ffi::OsString>);

    impl EnvGuard {
        fn set_cto_root(root: &std::path::Path) -> Self {
            let previous = std::env::var_os("CTO_AGENT_ROOT");
            unsafe {
                std::env::set_var("CTO_AGENT_ROOT", root);
            }
            Self(previous)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            if let Some(previous) = self.0.take() {
                unsafe {
                    std::env::set_var("CTO_AGENT_ROOT", previous);
                }
            } else {
                unsafe {
                    std::env::remove_var("CTO_AGENT_ROOT");
                }
            }
        }
    }

    #[test]
    fn queue_loop_interrupt_as_task_materializes_interrupt_and_task_consistently()
    -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("interrupt_materialize");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let interrupt_id = enqueue_loop_interrupt(
            &paths,
            "bios",
            "Michael Welsch",
            "Bitte priorisiere diese Eingabe im naechsten Supervisor-Tick.",
        )?;
        let task = queue_loop_interrupt_as_task(&paths, interrupt_id)?
            .expect("interrupt should materialize into a task");
        let task_reload = load_task_by_id(&paths, task.id)?.expect("task should reload");
        let interrupt =
            load_loop_interrupt_by_id(&paths, interrupt_id)?.expect("interrupt should still exist");

        assert_eq!(task_reload.source_interrupt_id, Some(interrupt_id));
        assert_eq!(task_reload.title, "BIOS-Chat");
        assert_eq!(interrupt.status, "queued");
        assert!(
            interrupt
                .response
                .unwrap_or_default()
                .contains(&format!("task {}", task.id)),
            "interrupt should reference the materialized task"
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn claim_next_worker_job_marks_job_running_before_returning() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("worker_claim");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        let now = now_iso();
        conn.execute(
            "INSERT INTO worker_jobs(
                created_at, updated_at, parent_task_id, parent_task_title, worker_kind,
                contract_title, contract_detail, status, request_note
             ) VALUES(?1, ?2, 41, 'Parent task', 'browser_agent', 'First contract', 'Run first job', 'queued', '')",
            params![now, now],
        )?;
        conn.execute(
            "INSERT INTO worker_jobs(
                created_at, updated_at, parent_task_id, parent_task_title, worker_kind,
                contract_title, contract_detail, status, request_note
             ) VALUES(?1, ?2, 42, 'Parent task', 'browser_agent', 'Second contract', 'Run second job', 'queued', '')",
            params![now_iso(), now_iso()],
        )?;
        drop(conn);

        let claimed = claim_next_worker_job(&paths)?.expect("worker job should be claimed");
        assert_eq!(claimed.status, "running");
        assert_eq!(claimed.contract_title, "First contract");

        let conn = open_db(&paths)?;
        let running_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM worker_jobs WHERE status = 'running'",
            [],
            |row| row.get(0),
        )?;
        let queued_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM worker_jobs WHERE status = 'queued'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(running_count, 1);
        assert_eq!(queued_count, 1);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn reprioritize_tasks_materializes_rows_before_updating() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("reprioritize");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        for (status, detail) in [
            ("queued", "Mail communication path for owner@example.com"),
            ("active", "Make the homepage and BIOS more visible"),
        ] {
            conn.execute(
                "INSERT INTO tasks(
                    created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                    title, detail, trust_level, priority_score, status
                 ) VALUES(?1, ?2, NULL, NULL, NULL, 'bios', 'Michael Welsch', 'channel_expansion',
                    'Assess communication paths', ?3, 'owner_trust', 0, ?4)",
                params![now_iso(), now_iso(), detail, status],
            )?;
        }
        drop(conn);

        reprioritize_tasks(&paths)?;

        let conn = open_db(&paths)?;
        let reprioritized_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM tasks WHERE priority_score > 0",
            [],
            |row| row.get(0),
        )?;
        let focus_mode: String = conn.query_row(
            "SELECT mode FROM focus_state WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?;
        let active_focus_task_id: Option<i64> = conn.query_row(
            "SELECT active_task_id FROM focus_state WHERE singleton = 1",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(reprioritized_count, 2);
        assert_eq!(focus_mode, "execute_task");
        assert!(active_focus_task_id.is_some());

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn queued_task_in_grosshirn_cooldown_is_skipped_until_cooldown_expires() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("queued-task-grosshirn-cooldown");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let first = enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Primary owner task",
            "Critical owner task that should cool down after grosshirn stalls.",
            1000,
        )?;
        let second = enqueue_internal_task(
            &paths,
            None,
            "root_trust",
            "Secondary task",
            "Another queued task.",
            120,
        )?;

        arm_task_grosshirn_boost(&paths, first.id, &first.title, "test boost", 1120)?;
        release_task_grosshirn_boost(&paths, first.id, "parking the task during cooldown")?;

        let queued = list_queued_tasks(&paths, 8)?;
        assert_eq!(queued.first().map(|task| task.id), Some(second.id));
        assert!(queued.iter().all(|task| task.id != first.id));
        assert!(task_is_in_grosshirn_cooldown(&paths, first.id));

        let conn = open_db(&paths)?;
        conn.execute(
            "UPDATE brain_routing_state
             SET cooldown_until = ?1
             WHERE singleton = 1",
            params![recent_past_iso(30)],
        )?;
        drop(conn);

        let queued_after = list_queued_tasks(&paths, 8)?;
        assert_eq!(queued_after.first().map(|task| task.id), Some(first.id));
        assert!(!task_is_in_grosshirn_cooldown(&paths, first.id));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn select_next_task_skips_grosshirn_cooldown_task() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("select-next-task-grosshirn-cooldown");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let first = enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Primary owner task",
            "Critical owner task that should cool down before selection.",
            1000,
        )?;
        let second = enqueue_internal_task(
            &paths,
            None,
            "root_trust",
            "Secondary task",
            "Another queued task.",
            120,
        )?;

        arm_task_grosshirn_boost(&paths, first.id, &first.title, "test boost", 1120)?;
        release_task_grosshirn_boost(&paths, first.id, "parking the task during cooldown")?;

        let selected = select_next_task(&paths)?.expect("a non-cooled task should be selected");
        assert_eq!(selected.id, second.id);
        assert_eq!(
            load_task_by_id(&paths, first.id)?
                .expect("cooled task should still exist")
                .status,
            "queued"
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn retryable_runtime_stall_requeues_blocked_tasks() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("retryable-runtime-stall-requeue");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let retryable = enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Retryable owner task",
            "The local runtime disappeared during execution.",
            1000,
        )?;
        block_task(
            &paths,
            retryable.id,
            "failed to connect to 127.0.0.1:1234",
            "The local endpoint vanished mid-turn.",
            None,
        )?;
        let retryable_blocked =
            load_task_by_id(&paths, retryable.id)?.expect("retryable task should be blocked");
        assert_eq!(retryable_blocked.status, "blocked");
        assert_eq!(
            retryable_blocked.last_checkpoint_summary.as_deref(),
            Some("failed to connect to 127.0.0.1:1234")
        );
        assert_eq!(
            list_task_checkpoints(&paths, retryable.id, 1)?
                .first()
                .map(|checkpoint| checkpoint.summary.as_str()),
            Some("failed to connect to 127.0.0.1:1234")
        );
        let conn = open_db(&paths)?;
        let retryable_match_count = conn.query_row(
            "SELECT COUNT(*)
             FROM tasks
             WHERE status = 'blocked'
               AND lower(COALESCE(last_checkpoint_summary, '')) LIKE '%failed to connect to 127.0.0.1%'",
            [],
            |row| row.get::<_, i64>(0),
        )?;
        assert_eq!(retryable_match_count, 1);
        let mut stmt = conn.prepare(
            "SELECT id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                    title, detail, trust_level, priority_score, status, run_count,
                    last_checkpoint_summary, last_checkpoint_at, last_output
             FROM tasks
             WHERE status = 'blocked'
               AND (
                    lower(COALESCE(last_checkpoint_summary, '')) LIKE '%failed to connect to 127.0.0.1%'
                    OR lower(COALESCE(last_checkpoint_summary, '')) LIKE '%connection refused%'
                    OR lower(COALESCE(last_checkpoint_summary, '')) LIKE '%tcp connect error%'
                    OR lower(COALESCE(last_checkpoint_summary, '')) LIKE '%os error 111%'
                    OR lower(COALESCE(last_checkpoint_summary, '')) LIKE '%workspace execution contract is active and no machine path ran in this turn%'
               )
             ORDER BY priority_score DESC, id ASC
             LIMIT ?1",
        )?;
        let matching = stmt
            .query_map(params![8_i64], map_task_row)?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        assert_eq!(
            matching.iter().map(|task| task.id).collect::<Vec<_>>(),
            vec![retryable.id]
        );

        let durable_block = enqueue_internal_task(
            &paths,
            None,
            "root_trust",
            "Durable block",
            "This task is blocked for a real task reason.",
            900,
        )?;
        block_task(
            &paths,
            durable_block.id,
            "Waiting for missing credentials.",
            "This should not be auto-requeued.",
            None,
        )?;

        let revived = requeue_retryable_blocked_tasks_after_runtime_stall(&paths, 8)?;
        assert_eq!(revived, vec![retryable.id]);

        let retryable_reload =
            load_task_by_id(&paths, retryable.id)?.expect("retryable task should reload");
        assert_eq!(retryable_reload.status, "queued");
        assert_eq!(
            retryable_reload.last_checkpoint_summary.as_deref(),
            Some("Retryable runtime stall cleared enough to retry this task.")
        );

        let durable_reload =
            load_task_by_id(&paths, durable_block.id)?.expect("durable task should reload");
        assert_eq!(durable_reload.status, "blocked");

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn selecting_a_different_task_clears_a_stale_grosshirn_boost() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("select-next-task-clears-stale-boost");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let first = enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Primary owner task",
            "The highest-priority task that should become active.",
            1500,
        )?;
        let second = enqueue_internal_task(
            &paths,
            None,
            "root_trust",
            "Secondary task",
            "A lower-priority task that previously held a stale grosshirn boost.",
            120,
        )?;

        arm_task_grosshirn_boost(&paths, second.id, &second.title, "test stale boost", 1120)?;

        let selected = select_next_task(&paths)?.expect("a task should be selected");
        assert_eq!(selected.id, first.id);

        let state = load_brain_routing_state(&paths)?;
        assert_eq!(state.route_mode, "kleinhirn");
        assert_eq!(state.boosted_task_id, None);
        assert!(
            state.last_deactivation_reason.contains("task #")
                && state
                    .last_deactivation_reason
                    .contains(&first.id.to_string())
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn parked_task_cooldown_survives_another_task_grosshirn_boost() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("parked-task-cooldown-survives-other-boost");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let first = enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Primary owner task",
            "Critical owner task that should remain cooled while other work borrows grosshirn.",
            1000,
        )?;
        let second = enqueue_internal_task(
            &paths,
            None,
            "owner_binding",
            "Secondary owner task",
            "Another task that may borrow grosshirn after the first task is parked.",
            380,
        )?;

        arm_task_grosshirn_boost(&paths, first.id, &first.title, "first boost", 1120)?;
        release_task_grosshirn_boost(&paths, first.id, "parking the first task during cooldown")?;
        arm_task_grosshirn_boost(&paths, second.id, &second.title, "second boost", 1120)?;

        assert!(task_is_in_grosshirn_cooldown(&paths, first.id));
        let selected = select_next_task(&paths)?.expect("the second task should be selected");
        assert_eq!(selected.id, second.id);
        assert_eq!(
            load_task_by_id(&paths, first.id)?
                .expect("parked task should still exist")
                .status,
            "queued"
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn queued_owner_interrupts_with_equal_priority_prefer_newest_first() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("queued-owner-interrupt-newest-first");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let older_interrupt = enqueue_loop_interrupt(
            &paths,
            "bios",
            "Michael Welsch",
            "Older owner interrupt.",
        )?;
        let older = queue_loop_interrupt_as_task(&paths, older_interrupt)?
            .expect("older owner interrupt should materialize as a task");
        let newer_interrupt = enqueue_loop_interrupt(
            &paths,
            "attach_terminal",
            "Michael Welsch",
            "Newer owner interrupt.",
        )?;
        let newer = queue_loop_interrupt_as_task(&paths, newer_interrupt)?
            .expect("newer owner interrupt should materialize as a task");

        let queued = list_queued_tasks(&paths, 8)?;
        assert_eq!(queued.first().map(|task| task.id), Some(newer.id));
        assert_eq!(queued.get(1).map(|task| task.id), Some(older.id));

        let selected = select_next_task(&paths)?.expect("newest owner task should be selected");
        assert_eq!(selected.id, newer.id);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn context_preparation_inherits_parent_priority_during_reprioritization() -> anyhow::Result<()>
    {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("reprioritize-context-prep");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        conn.execute(
            "INSERT INTO tasks(
                created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status
             ) VALUES(?1, ?2, NULL, NULL, NULL, 'attach_terminal', 'Michael Welsch', 'owner_interrupt',
                'Build the C++ app', 'Direct owner task', 'system', 1000, 'queued')",
            params![now_iso(), now_iso()],
        )?;
        let parent_id = conn.last_insert_rowid();
        conn.execute(
            "INSERT INTO tasks(
                created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status
             ) VALUES(?1, ?2, ?3, NULL, NULL, 'system_guard', 'system_guard', 'context_preparation',
                'Prepare context for task', 'Preparation step', 'system', 40, 'queued')",
            params![now_iso(), now_iso(), parent_id],
        )?;
        drop(conn);

        reprioritize_tasks(&paths)?;

        let conn = open_db(&paths)?;
        let prep_priority: i64 = conn.query_row(
            "SELECT priority_score FROM tasks WHERE task_kind = 'context_preparation' LIMIT 1",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(prep_priority, 1005);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn context_preparation_ready_requeue_resets_parent_run_count() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("context-ready-reset-run-count");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        conn.execute(
            "INSERT INTO tasks(
                created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count
             ) VALUES(?1, ?2, NULL, NULL, NULL, 'attach_terminal', 'Michael Welsch', 'owner_interrupt',
                'Build the C++ app', 'Direct owner task', 'system', 1000, 'queued', 9)",
            params![now_iso(), now_iso()],
        )?;
        let task_id = conn.last_insert_rowid();
        drop(conn);

        requeue_task_with_checkpoint_kind(
            &paths,
            task_id,
            "context_preparation_ready",
            "Fresh context is ready.",
            "The task received a fresh execution handoff.",
            None,
        )?;

        let conn = open_db(&paths)?;
        let run_count: i64 = conn.query_row(
            "SELECT run_count FROM tasks WHERE id = ?1",
            params![task_id],
            |row| row.get(0),
        )?;
        let status: String = conn.query_row(
            "SELECT status FROM tasks WHERE id = ?1",
            params![task_id],
            |row| row.get(0),
        )?;
        assert_eq!(run_count, 0);
        assert_eq!(status, "queued");

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn explicit_bios_grosshirn_switch_is_classified_as_activation() {
        let task_kind = classify_task_kind(
            "Wechsle jetzt auf GPT-5.4 als Grosshirn und konfiguriere den OpenAI API Token fuer den Moduswechsel.",
        );
        assert_eq!(task_kind, "grosshirn_activation");
    }

    #[test]
    fn explicit_bios_local_model_switch_is_classified_separately() {
        let task_kind =
            classify_task_kind("Wechsle jetzt lokal auf Qwen3.5-35B-A3B als Kleinhirn.");
        assert_eq!(task_kind, "local_model_switch");
    }

    #[test]
    fn owner_attach_interrupt_is_not_reclassified_into_self_preservation() {
        let task_kind = classify_interrupt_task_kind(
            "Michael Welsch",
            "attach_terminal",
            "Michael Welsch",
            "Stop self-preservation meta unless the loop is actually unhealthy and read the repo now.",
        );
        assert_eq!(task_kind, "owner_interrupt");
    }

    #[test]
    fn bios_chat_interrupt_is_not_reclassified_from_chat_content() {
        let task_kind = classify_interrupt_task_kind(
            "Michael Welsch",
            "bios",
            "Michael Welsch",
            "Wechsle jetzt auf GPT-5.4 und konfiguriere sofort den API Key.",
        );
        assert_eq!(task_kind, "owner_interrupt");
    }

    #[test]
    fn bios_chat_signal_kind_does_not_scan_message_keywords() {
        let signal_kind = classify_turn_signal_kind(
            "Michael Welsch",
            "bios",
            "Michael Welsch",
            "Das ist nicht dringend, nicht kritisch und trotzdem nur ein normaler BIOS-Chat.",
        );
        assert_eq!(signal_kind, "interrupt");
    }

    #[test]
    fn attach_chat_priority_ignores_urgency_keywords() {
        let loop_safety = default_loop_safety_policy();
        let calm = compute_priority_score(
            &loop_safety,
            "Michael Welsch",
            "owner_interrupt",
            "attach_terminal",
            "Michael Welsch",
            "Bitte schaue spaeter drauf.",
        );
        let urgent = compute_priority_score(
            &loop_safety,
            "Michael Welsch",
            "owner_interrupt",
            "attach_terminal",
            "Michael Welsch",
            "Bitte jetzt sofort dringend kritisch anschauen.",
        );
        assert_eq!(urgent, calm);
    }

    #[test]
    fn owner_email_interrupt_with_address_stays_owner_interrupt() {
        let task_kind = classify_interrupt_task_kind(
            "Michael Welsch",
            "email",
            "Michael Welsch <michael.welsch@metric-space.ai>",
            "Stop writing the same mail again and again.",
        );
        assert_eq!(task_kind, "owner_interrupt");
    }

    #[test]
    fn first_email_sync_run_is_treated_as_baseline() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("email-sync-baseline");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        conn.execute_batch(include_str!("../scripts/communication_schema.sql"))?;
        drop(conn);

        assert!(communication_email_sync_needs_baseline(&paths)?);

        let conn = open_db(&paths)?;
        conn.execute(
            "INSERT INTO communication_sync_runs(
                run_key, channel, account_key, folder_hint, started_at, finished_at,
                ok, fetched_count, stored_count, error_text, metadata_json
             ) VALUES(?1, 'email', 'cto1@metric-space.ai', 'INBOX', ?2, ?2, 1, 3, 3, '', '{}')",
            params!["run-1", now_iso()],
        )?;
        drop(conn);

        assert!(!communication_email_sync_needs_baseline(&paths)?);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn repo_root_instruction_is_not_misclassified_as_root_trust() {
        let task_kind = classify_task_kind(
            "Dringende Chefanweisung: Der Repo-Root wurde bereits erfolgreich gelesen. Wiederhole ls -1 nicht. Lies stattdessen den Anfang der README.",
        );
        assert_eq!(task_kind, "owner_interrupt");
    }

    #[test]
    fn owner_speaker_gets_owner_override_even_without_owner_name() {
        let score = compute_priority_score(
            &default_loop_safety_policy(),
            "",
            "grosshirn_activation",
            "bios",
            "owner",
            "Wechsle jetzt in den Grosshirn-Modus.",
        );
        assert!(score >= 1000, "expected owner override floor, got {score}");
    }

    #[test]
    fn grosshirn_activation_outranks_generic_owner_model_request() {
        let loop_safety = default_loop_safety_policy();
        let activation = compute_priority_score(
            &loop_safety,
            "",
            "grosshirn_activation",
            "bios",
            "owner",
            "Wechsle jetzt auf GPT-5.4 als Grosshirn.",
        );
        let generic = compute_priority_score(
            &loop_safety,
            "",
            "model_or_resource",
            "bios",
            "owner",
            "Pruefe jetzt sofort den Grosshirn-Ausfallpfad.",
        );
        assert!(
            activation > generic,
            "expected grosshirn activation ({activation}) to outrank generic model task ({generic})"
        );
    }

    #[test]
    fn review_continue_requeues_parent_without_automatic_grosshirn_boost() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("review_grosshirn_boost");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;
        std::fs::write(
            paths.runtime_dir.join("kleinhirn.env"),
            "\
CTO_AGENT_KLEINHIRN_BASE_URL=http://127.0.0.1:1234/v1\n\
CTO_AGENT_KLEINHIRN_RUNTIME_MODEL=openai/gpt-oss-20b\n\
CTO_AGENT_GROSSHIRN_API_KEY=test-grosshirn\n\
CTO_AGENT_GROSSHIRN_MODEL=gpt-5.4\n\
CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER=openai_responses\n\
CTO_AGENT_GROSSHIRN_BASE_URL=https://api.openai.com/v1\n",
        )?;

        let conn = open_db(&paths)?;
        let now = now_iso();
        conn.execute(
            "INSERT INTO tasks(
                id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count
            ) VALUES(?1, ?2, ?3, NULL, NULL, NULL, 'bios', 'owner', 'workspace_repair',
                'Complex repair', 'The kleinhirn did not get through this cleanly.', 'owner_trust', 400, 'await_review', 3)",
            params![77_i64, now, now],
        )?;
        conn.execute(
            "INSERT INTO tasks(
                id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count
            ) VALUES(?1, ?2, ?3, ?4, NULL, NULL, 'system_guard', 'self_review', 'self_review',
                'Self-review task #77 before completion', 'Check the real state.', 'system', 890, 'queued', 0)",
            params![78_i64, now, now, 77_i64],
        )?;
        drop(conn);

        let review = load_task_by_id(&paths, 78)?.expect("review task");
        complete_review_task(
            &paths,
            &review,
            "continue",
            "The task could not be solved cleanly with kleinhirn.",
            "The current kleinhirn was overloaded; the next bounded step temporarily needs grosshirn.",
            None,
        )?;

        let parent = load_task_by_id(&paths, 77)?.expect("parent task");
        assert_eq!(parent.status, "queued");
        assert!(!task_has_active_grosshirn_boost(&paths, 77));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn enqueue_internal_task_reuses_recently_blocked_system_guard_task() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("blocked_internal_task_reuse");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let first = enqueue_internal_task(
            &paths,
            None,
            "homepage_bridge",
            "Secure homepage and BIOS",
            "First mandatory task.",
            880,
        )?;
        block_task(
            &paths,
            first.id,
            "Browser verification is still missing.",
            "The browser artifacts are still missing.",
            None,
        )?;

        let reused = enqueue_internal_task(
            &paths,
            None,
            "homepage_bridge",
            "Secure homepage and BIOS",
            "The same mandatory task must not be spawned immediately as a duplicate.",
            880,
        )?;

        assert_eq!(reused.id, first.id);
        assert_eq!(reused.status, "blocked");

        let queued = list_queued_tasks(&paths, 32)?;
        assert!(
            !queued
                .iter()
                .any(|task| task.id != first.id && task.task_kind == "homepage_bridge")
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn orphaned_await_review_parent_is_requeued() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("orphaned_review_wait");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        let now = now_iso();
        conn.execute(
            "INSERT INTO tasks(
                id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count
            ) VALUES(?1, ?2, ?3, NULL, NULL, NULL, 'bios', 'owner', 'workspace_repair',
                'Complete the review recovery', 'The task was left behind in await_review.', 'owner_trust', 420, 'await_review', 2)",
            params![91_i64, now, now],
        )?;
        drop(conn);

        let recovered = recover_orphaned_review_waits(&paths)?;
        assert_eq!(recovered, vec![91_i64]);

        let parent = load_task_by_id(&paths, 91)?.expect("parent task");
        assert_eq!(parent.status, "queued");
        assert_eq!(
            parent.last_checkpoint_summary.as_deref(),
            Some("Review wait recovered for task #91 Complete the review recovery.")
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn owner_interrupt_host_change_without_execution_evidence_fails_completion_gate()
    -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("owner_interrupt_gate_fail");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let task = TaskRecord {
            id: 501,
            created_at: now_iso(),
            updated_at: now_iso(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: "Stelle jetzt das Tastaturlayout auf Deutsch um".to_string(),
            detail: "Carry out the change on the real host and verify it.".to_string(),
            trust_level: "owner_trust".to_string(),
            priority_score: 1000,
            status: "queued".to_string(),
            run_count: 1,
            last_checkpoint_summary: Some(
                "Need targeted history/state before changing keyboard settings again.".to_string(),
            ),
            last_checkpoint_at: Some(now_iso()),
            last_output: Some(
                "I need more context before issuing another system change.".to_string(),
            ),
        };

        let failure = task_completion_gate_failure(
            &paths,
            &task,
            Some("looks complete"),
            Some("Need targeted history/state before changing keyboard settings again."),
        )
        .expect("owner interrupt should fail gate");
        assert!(failure.contains("real host mutation"));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn owner_interrupt_host_change_with_execution_evidence_passes_completion_gate()
    -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("owner_interrupt_gate_pass");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let task = TaskRecord {
            id: 502,
            created_at: now_iso(),
            updated_at: now_iso(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: "Stelle jetzt das Tastaturlayout auf Deutsch um".to_string(),
            detail: "Carry out the change on the real host and verify it.".to_string(),
            trust_level: "owner_trust".to_string(),
            priority_score: 1000,
            status: "queued".to_string(),
            run_count: 1,
            last_checkpoint_summary: Some(
                "Keyboard layout changed to de and verified via setxkbmap/localectl.".to_string(),
            ),
            last_checkpoint_at: Some(now_iso()),
            last_output: Some(
                "Changed keyboard layout to German with setxkbmap de and verified layout=de."
                    .to_string(),
            ),
        };

        let failure = task_completion_gate_failure(
            &paths,
            &task,
            Some("done"),
            Some("Changed keyboard layout to German with setxkbmap de and verified layout=de."),
        );
        assert!(failure.is_none());

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn promoted_learning_entries_refresh_learning_working_set() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("learning_working_set");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        let now = now_iso();
        conn.execute(
            "INSERT INTO tasks(
                id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count
            ) VALUES(?1, ?2, ?3, NULL, NULL, NULL, 'system_guard', 'system_guard', 'tool_exploration',
                'Inspect tool path', 'Test tool boundaries deliberately.', 'system', 240, 'queued', 0)",
            params![91_i64, now, now],
        )?;
        drop(conn);

        let task = load_task_by_id(&paths, 91)?.expect("task should exist");
        store_learning_entries(
            &paths,
            &task,
            7,
            &[
                LearningEntryDraft {
                    learning_class: "operational".to_string(),
                    summary: "Check real runtime and host limits before risky tool work.".to_string(),
                    detail: "Before any deeper tool path, the agent should explicitly inspect runtime, host, and side effects.".to_string(),
                    evidence: "The bounded tool test showed that claimed capability can easily be wrong without runtime verification.".to_string(),
                    applicability: "Before tool bootstrap, browser setup, and exec work.".to_string(),
                    confidence: 0.9,
                    salience: 90,
                },
                LearningEntryDraft {
                    learning_class: "negative".to_string(),
                    summary: "Do not infer tool capability from contracts or wishful thinking.".to_string(),
                    detail: "Capabilities must be checked against reality instead of being assumed from policy or prompt text.".to_string(),
                    evidence: "The task was tool exploration; real verification was necessary to avoid false confidence.".to_string(),
                    applicability: "On browser, exec, and repair paths.".to_string(),
                    confidence: 0.85,
                    salience: 82,
                },
            ],
            "candidate",
            "test_seed",
        )?;
        assert!(load_memory_summary(&paths, "learning_working_set")?.is_none());

        let promoted = promote_learning_candidates_for_task(&paths, 91, "test_promote")?;
        assert_eq!(promoted, 2);

        let active = list_active_learning_entries(&paths, 8)?;
        assert_eq!(active.len(), 2);
        let working_set = load_memory_summary(&paths, "learning_working_set")?
            .expect("working set summary should exist");
        assert!(working_set.contains("[ops]"));
        assert!(working_set.contains("[neg]"));
        let operational = load_memory_summary(&paths, "learning_operational")?
            .expect("operational summary should exist");
        assert!(operational.contains("Check real runtime and host limits"));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn person_paths_capture_conversation_and_learning_refs() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("person_learning_path");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let profile = record_person_interaction(
            &paths,
            "bios",
            "Michael Welsch",
            "Bitte erinnere dich daran, dass ich fruehere Gespraeche nicht wiederholen will.",
            "test_interrupt:1",
        )?
        .expect("person profile should be created");
        assert_eq!(profile.display_name, "Michael Welsch");

        let conn = open_db(&paths)?;
        let now = now_iso();
        conn.execute(
            "INSERT INTO tasks(
                id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count
             ) VALUES(?1, ?2, ?3, NULL, NULL, NULL, 'bios', 'Michael Welsch', 'owner_interrupt',
                'Remember owner priority', 'Keep owner preferences recorded reliably.', 'owner_trust', 500, 'queued', 0)",
            params![92_i64, now, now],
        )?;
        drop(conn);

        let task = load_task_by_id(&paths, 92)?.expect("task should exist");
        store_learning_entries(
            &paths,
            &task,
            8,
            &[LearningEntryDraft {
                learning_class: "operational".to_string(),
                summary: "In recurring conversations, person preferences must stay active in recall.".to_string(),
                detail: "Owner and person preferences must not live only in raw prose; they must remain referencable in the learning path and people path.".to_string(),
                evidence: "The speaker explicitly stressed that repeating earlier conversations is frustrating.".to_string(),
                applicability: "For BIOS, homepage, mail, and later proactive decisions.".to_string(),
                confidence: 0.92,
                salience: 91,
            }],
            "active",
            "test_person_learning",
        )?;

        let people_working_set = load_memory_summary(&paths, "people_working_set")?
            .expect("people working set should exist");
        assert!(people_working_set.contains("Michael Welsch"));

        let notes = list_person_notes_for_person(&paths, profile.id, 8)?;
        assert!(notes.iter().any(|note| note.note_kind == "conversation"));
        assert!(notes.iter().any(|note| note.note_kind == "learning_ref"));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn proactive_contact_validation_updates_candidate_and_notebook() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("proactive_contact_validation");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        let now = now_iso();
        conn.execute(
            "INSERT INTO tasks(
                id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count
             ) VALUES(?1, ?2, ?3, NULL, NULL, NULL, 'system_guard', 'system_guard', 'person_relationship_review',
                'Maintain people paths', 'Derive helpful suggestions for people.', 'system', 230, 'queued', 0)",
            params![93_i64, now, now],
        )?;
        drop(conn);

        let task = load_task_by_id(&paths, 93)?.expect("task should exist");
        let candidate = store_proactive_contact_candidate(
            &paths,
            &task,
            9,
            &ProactiveContactDraft {
                person_name: "Michael Welsch".to_string(),
                person_email: "".to_string(),
                channel: "bios".to_string(),
                subject: "Proposal for clearer priorities".to_string(),
                body: "I derived a clearer prioritization for the next CTO steps from the recent conversations.".to_string(),
                rationale: "That reduces repetition and visibly incorporates earlier conversations.".to_string(),
                conflict_check: "No obvious conflict of interest; the proposal only organizes already stated priorities.".to_string(),
            },
            true,
            "test_draft",
        )?;
        let _ = attach_validation_task_to_candidate(&paths, candidate.id, 501)?;
        let validated = apply_proactive_contact_validation(
            &paths,
            501,
            &ProactiveContactValidationDraft {
                decision: "approve".to_string(),
                note: "The proposal serves the person's interest and does not conflict with known signals.".to_string(),
                revised_subject: "".to_string(),
                revised_body: "".to_string(),
            },
            "test_validation",
        )?
        .expect("candidate should be found by validation task");
        assert_eq!(validated.status, "approved");

        let people = list_person_profiles(&paths, 4)?;
        assert!(people.iter().any(|profile| {
            profile.display_name == "Michael Welsch"
                && profile
                    .notebook_summary
                    .contains("Approved proactive proposal")
        }));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn proactive_contact_dispatch_result_updates_person_memory() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("proactive_contact_dispatch");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let conn = open_db(&paths)?;
        let now = now_iso();
        conn.execute(
            "INSERT INTO tasks(
                id, created_at, updated_at, parent_task_id, worker_job_id, source_interrupt_id, source_channel, speaker, task_kind,
                title, detail, trust_level, priority_score, status, run_count
             ) VALUES(?1, ?2, ?3, NULL, NULL, NULL, 'system_guard', 'system_guard', 'person_relationship_review',
                'Maintain people paths', 'Derive helpful suggestions for people.', 'system', 230, 'queued', 0)",
            params![93_i64, now, now],
        )?;
        drop(conn);

        let task = load_task_by_id(&paths, 93)?.expect("task should exist");
        let candidate = store_proactive_contact_candidate(
            &paths,
            &task,
            11,
            &ProactiveContactDraft {
                person_name: "Michael Welsch".to_string(),
                person_email: "michael.welsch@metric-space.ai".to_string(),
                channel: "email".to_string(),
                subject: "Short proposal for the next CTO cycle".to_string(),
                body: "I derived a focused proposal for the next steps from the latest signals."
                    .to_string(),
                rationale:
                    "That saves follow-up questions and visibly incorporates earlier conversations."
                        .to_string(),
                conflict_check: "No visible conflict of interest.".to_string(),
            },
            true,
            "test_draft",
        )?;
        let _ = attach_dispatch_task_to_candidate(&paths, candidate.id, 601)?;
        let updated = record_proactive_contact_dispatch_result(
            &paths,
            601,
            "sent",
            "email",
            "Local SMTP delivery succeeded.",
            "<msg-123@example.test>",
            "test_dispatch",
        )?
        .expect("candidate should be found by dispatch task");
        assert_eq!(updated.status, "sent");
        assert_eq!(updated.dispatch_channel, "email");
        assert_eq!(updated.outbound_message_id, "<msg-123@example.test>");

        let people = list_person_profiles(&paths, 4)?;
        assert!(people.iter().any(|profile| {
            profile.display_name == "Michael Welsch"
                && profile
                    .conversation_memory_summary
                    .contains("Contacted proactively via email")
                && profile.notebook_summary.contains("Proactive message sent")
        }));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn maintenance_mode_oscillation_stays_out_of_public_event_feed() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("maintenance-mode-events");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        set_focus_state(
            &paths,
            "observe",
            None,
            "",
            "Observe current signals, resources and queued work before reprioritization.",
        )?;
        set_focus_state(
            &paths,
            "reprioritize",
            None,
            "",
            "Task priorities recalculated.",
        )?;

        let conn = open_db(&paths)?;
        let published_events: i64 = conn.query_row(
            "SELECT COUNT(*) FROM agent_events WHERE method = 'mode/changed'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(published_events, 0);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn task_bound_maintenance_oscillation_stays_out_of_public_event_feed() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("task-bound-maintenance-mode-events");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        set_focus_state(
            &paths,
            "observe",
            Some(26),
            "Prepare context for task 25",
            "Observe current signals, resources and queued work before reprioritization.",
        )?;
        set_focus_state(
            &paths,
            "reprioritize",
            Some(26),
            "Prepare context for task 25",
            "Task priorities recalculated.",
        )?;

        let conn = open_db(&paths)?;
        let published_events: i64 = conn.query_row(
            "SELECT COUNT(*) FROM agent_events WHERE method = 'mode/changed'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(published_events, 0);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn task_attached_mode_change_is_still_published() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("task-mode-events");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        set_focus_state(
            &paths,
            "await_review",
            Some(42),
            "Review context draft",
            "A review-ready task became visible.",
        )?;

        let conn = open_db(&paths)?;
        let published_events: i64 = conn.query_row(
            "SELECT COUNT(*) FROM agent_events WHERE method = 'mode/changed'",
            [],
            |row| row.get(0),
        )?;
        assert_eq!(published_events, 1);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn continue_active_task_keeps_status_active_and_focus_attached() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("continue-active-task");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Build the repo task",
            "Keep the substantive owner task active across turns.",
            1000,
        )?;
        let selected = select_next_task(&paths)?.expect("task should activate");
        let prior_run_count = selected.run_count;

        continue_active_task(
            &paths,
            selected.id,
            "execute_task",
            "One bounded repo step completed with new evidence.",
            "Verified file edit and build output.",
            Some("machine evidence"),
        )?;

        let reloaded =
            load_task_by_id(&paths, selected.id)?.expect("continued task should still exist");
        let focus = load_focus_state(&paths)?;

        assert_eq!(reloaded.status, "active");
        assert_eq!(reloaded.run_count, prior_run_count + 1);
        assert_eq!(focus.mode, "execute_task");
        assert_eq!(focus.active_task_id, Some(selected.id));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }
}

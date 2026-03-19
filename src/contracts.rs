use crate::storage::append_jsonl;
use crate::storage::load_json;
use crate::storage::load_jsonl;
use crate::storage::save_json;
use anyhow::Context;
use chrono::Utc;
use getrandom::getrandom;
use hex::encode;
use pbkdf2::pbkdf2_hmac_array;
use serde::Deserialize;
use serde::Serialize;
use sha2::Digest;
use sha2::Sha256;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

const PASSWORD_ITERATIONS: u32 = 200_000;

#[derive(Debug, Clone)]
pub struct Paths {
    pub root: PathBuf,
    pub contracts_dir: PathBuf,
    pub runtime_dir: PathBuf,
    pub uploads_dir: PathBuf,
    pub browser_artifacts_dir: PathBuf,
    pub recovery_dir: PathBuf,
    pub history_dir: PathBuf,
    pub models_dir: PathBuf,
    pub homepage_dir: PathBuf,
    pub bootstrap_dir: PathBuf,
    pub context_dir: PathBuf,
    pub system_dir: PathBuf,
    pub browser_dir: PathBuf,
    pub genome_path: PathBuf,
    pub bios_path: PathBuf,
    pub org_path: PathBuf,
    pub root_auth_path: PathBuf,
    pub model_policy_path: PathBuf,
    pub homepage_policy_path: PathBuf,
    pub bootstrap_task_pack_path: PathBuf,
    pub installation_bootstrap_path: PathBuf,
    pub context_policy_path: PathBuf,
    pub context_governance_policy_path: PathBuf,
    pub mode_system_policy_path: PathBuf,
    pub loop_safety_policy_path: PathBuf,
    pub execution_authority_policy_path: PathBuf,
    pub browser_engine_policy_path: PathBuf,
    pub browser_capability_policy_path: PathBuf,
    pub browser_subworker_policy_path: PathBuf,
    pub self_preservation_state_path: PathBuf,
    pub origin_story_path: PathBuf,
    pub creation_ledger_path: PathBuf,
    pub boot_log_path: PathBuf,
    pub agent_state_path: PathBuf,
    pub system_census_path: PathBuf,
    pub browser_engine_state_path: PathBuf,
    pub runtime_db_path: PathBuf,
    pub attach_socket_path: PathBuf,
    pub runtime_lock_path: PathBuf,
    pub pending_hard_reset_report_path: PathBuf,
    pub certs_dir: PathBuf,
    pub tls_cert_path: PathBuf,
    pub tls_key_path: PathBuf,
}

impl Paths {
    pub fn discover() -> anyhow::Result<Self> {
        let root = std::env::var("CTO_AGENT_ROOT")
            .map(PathBuf::from)
            .map(Ok)
            .unwrap_or_else(|_| std::env::current_dir())
            .context("failed to resolve CTO-Agent root directory")?;
        let contracts_dir = root.join("contracts");
        let history_dir = contracts_dir.join("history");
        let models_dir = contracts_dir.join("models");
        let homepage_dir = contracts_dir.join("homepage");
        let bootstrap_dir = contracts_dir.join("bootstrap");
        let context_dir = contracts_dir.join("context");
        let system_dir = contracts_dir.join("system");
        let browser_dir = contracts_dir.join("browser");
        let runtime_dir = root.join("runtime");
        let uploads_dir = runtime_dir.join("uploads");
        let browser_artifacts_dir = runtime_dir.join("browser");
        let recovery_dir = runtime_dir.join("recovery");
        let certs_dir = runtime_dir.join("certs");
        let state_dir = runtime_dir.join("state");
        let attach_socket_path = std::env::temp_dir().join(format!(
            "cto-agent-{}.sock",
            short_runtime_socket_key(&root)
        ));

        Ok(Self {
            root,
            contracts_dir: contracts_dir.clone(),
            runtime_dir: runtime_dir.clone(),
            uploads_dir: uploads_dir.clone(),
            browser_artifacts_dir: browser_artifacts_dir.clone(),
            recovery_dir: recovery_dir.clone(),
            history_dir: history_dir.clone(),
            models_dir: models_dir.clone(),
            homepage_dir: homepage_dir.clone(),
            bootstrap_dir: bootstrap_dir.clone(),
            context_dir: context_dir.clone(),
            system_dir: system_dir.clone(),
            browser_dir: browser_dir.clone(),
            genome_path: contracts_dir.join("genome/genome.json"),
            bios_path: contracts_dir.join("bios/bios.json"),
            org_path: contracts_dir.join("org/organigram.json"),
            root_auth_path: contracts_dir.join("root_auth/root_auth.json"),
            model_policy_path: models_dir.join("model-policy.json"),
            homepage_policy_path: homepage_dir.join("homepage-policy.json"),
            bootstrap_task_pack_path: bootstrap_dir.join("bootstrap-task-pack.json"),
            installation_bootstrap_path: bootstrap_dir.join("installation-bootstrap.json"),
            context_policy_path: context_dir.join("context-policy.json"),
            context_governance_policy_path: context_dir.join("context-governance-policy.json"),
            mode_system_policy_path: system_dir.join("mode-system-policy.json"),
            loop_safety_policy_path: system_dir.join("loop-safety-policy.json"),
            execution_authority_policy_path: system_dir.join("execution-authority-policy.json"),
            browser_engine_policy_path: browser_dir.join("browser-engine-policy.json"),
            browser_capability_policy_path: browser_dir.join("browser-capability-policy.json"),
            browser_subworker_policy_path: browser_dir.join("browser-subworker-policy.json"),
            self_preservation_state_path: system_dir.join("self-preservation-state.json"),
            origin_story_path: history_dir.join("origin-story.md"),
            creation_ledger_path: history_dir.join("creation-ledger.md"),
            boot_log_path: runtime_dir.join("boot_log.jsonl"),
            agent_state_path: state_dir.join("agent_state.json"),
            system_census_path: state_dir.join("system_census.json"),
            browser_engine_state_path: state_dir.join("browser_engine_state.json"),
            runtime_db_path: runtime_dir.join("cto_agent.db"),
            attach_socket_path,
            runtime_lock_path: runtime_dir.join("cto-agent.lock"),
            pending_hard_reset_report_path: recovery_dir.join("pending-hard-reset-report.json"),
            certs_dir: certs_dir.clone(),
            tls_cert_path: certs_dir.join("localhost.crt"),
            tls_key_path: certs_dir.join("localhost.key"),
        })
    }

    pub fn ensure_dirs(&self) -> anyhow::Result<()> {
        for dir in [
            self.contracts_dir.join("genome"),
            self.contracts_dir.join("bios"),
            self.contracts_dir.join("org"),
            self.contracts_dir.join("root_auth"),
            self.history_dir.clone(),
            self.models_dir.clone(),
            self.homepage_dir.clone(),
            self.bootstrap_dir.clone(),
            self.context_dir.clone(),
            self.system_dir.clone(),
            self.browser_dir.clone(),
            self.runtime_dir.join("state"),
            self.uploads_dir.clone(),
            self.browser_artifacts_dir.clone(),
            self.recovery_dir.clone(),
            self.certs_dir.clone(),
        ] {
            fs::create_dir_all(&dir)
                .with_context(|| format!("failed to create {}", dir.display()))?;
        }
        Ok(())
    }
}

fn short_runtime_socket_key(root: &Path) -> String {
    let digest = Sha256::digest(root.display().to_string().as_bytes());
    encode(digest)[..16].to_string()
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Genome {
    pub name: String,
    pub version: u32,
    pub principles: Vec<String>,
    pub immutable_genes: Vec<String>,
    pub adaptive_surfaces: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentIdentity {
    pub agent_name: String,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct OwnerRef {
    pub name: String,
    pub email: String,
    pub role: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Subordinates {
    pub people: Vec<String>,
    pub agents: Vec<String>,
    pub vendors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommunicationPolicy {
    pub primary_bootstrap_surface: String,
    pub allowed_channels: Vec<String>,
    pub future_channels: Vec<String>,
    pub preferred_identity_binding_surface: String,
    pub low_trust_redirect_surface: String,
    pub channel_rules: Vec<CommunicationChannelRule>,
    pub action_rules: Vec<ActionAuthorityRule>,
    pub escalation_policy: CommunicationEscalationPolicy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommunicationChannelRule {
    pub channel: String,
    pub layer: String,
    pub trust_level: String,
    pub interpretation: String,
    pub binding_power: String,
    pub sensitive_topic_policy: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CommunicationEscalationPolicy {
    pub sender_identity_checked_against_organigram: bool,
    pub unknown_sender_defaults_to_low_trust: bool,
    pub low_trust_channels_must_not_set_root_trust: bool,
    pub low_trust_channels_must_not_lock_branding: bool,
    pub may_refuse_sensitive_topics_on_low_trust_channels: bool,
    pub sensitive_topics_redirect_to: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ActionAuthorityRule {
    pub action_class: String,
    pub description: String,
    pub allowed_channels: Vec<String>,
    pub requires_root_verification: bool,
    pub requires_bios_primary: bool,
    pub terminal_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootAuthPolicy {
    pub superpassword_required: bool,
    pub set_via_website_only: bool,
    pub configured: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChangeRules {
    pub self_edit_allowed: bool,
    pub owner_or_board_required_for_upward_changes: bool,
    pub cto_may_edit_subtree_only: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Bios {
    pub name: String,
    pub version: u32,
    pub frozen: bool,
    pub frozen_at: Option<String>,
    pub presented_on_web: bool,
    pub website_path: String,
    pub agent_identity: AgentIdentity,
    pub mission: String,
    pub owner: OwnerRef,
    pub root_authorities: Vec<String>,
    pub reports_to: String,
    pub board: Vec<String>,
    pub peer_cxos: Vec<String>,
    pub subordinates: Subordinates,
    pub communication_policy: CommunicationPolicy,
    pub root_auth: RootAuthPolicy,
    pub change_rules: ChangeRules,
    pub boot_testimony_count: usize,
    pub last_drafted_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Organigram {
    pub owner: OwnerRef,
    pub reports_to: String,
    pub ceo: String,
    pub board: Vec<String>,
    pub peer_cxos: Vec<String>,
    pub subordinates: Subordinates,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RootAuthState {
    pub configured: bool,
    pub set_at: Option<String>,
    pub password_hash: Option<String>,
    pub salt: Option<String>,
    pub iterations: Option<u32>,
    pub last_verified_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AgentState {
    pub agent_name: String,
    pub bios_frozen: bool,
    pub owner_known: bool,
    pub reports_to_known: bool,
    pub supervisor_status: String,
    pub last_heartbeat_at: Option<String>,
    pub uptime_seconds: u64,
    #[serde(default)]
    pub active_turn_id: Option<i64>,
    #[serde(default)]
    pub active_turn_started_at: Option<String>,
    #[serde(default)]
    pub last_turn_completed_at: Option<String>,
    #[serde(default)]
    pub loop_health: String,
    #[serde(default)]
    pub unhealthy_reason: Option<String>,
    #[serde(default)]
    pub browser_engine_status: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct SystemCensus {
    pub captured_at: Option<String>,
    pub hostname: Option<String>,
    pub platform: Option<String>,
    pub agent_version: Option<String>,
    pub cpu_threads: Option<usize>,
    pub total_memory_gb: Option<u64>,
    #[serde(default)]
    pub gpu_count: Option<usize>,
    #[serde(default)]
    pub total_gpu_memory_gb: Option<u64>,
    #[serde(default)]
    pub max_single_gpu_memory_gb: Option<u64>,
    #[serde(default)]
    pub gpus: Option<Vec<GpuDevice>>,
    #[serde(default)]
    pub model_tune_candidates: Option<Vec<ModelTuneCandidate>>,
    pub cwd: Option<String>,
    pub pid: Option<u32>,
    pub top_level_entries: Option<Vec<String>>,
    #[serde(default)]
    pub desktop_session: Option<String>,
    #[serde(default)]
    pub chrome_binary: Option<String>,
    #[serde(default)]
    pub chrome_version: Option<String>,
    #[serde(default)]
    pub browser_engine_status: Option<String>,
    #[serde(default)]
    pub browser_headless_ready: Option<bool>,
    #[serde(default)]
    pub browser_interactive_ready: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct GpuDevice {
    pub index: usize,
    pub name: String,
    pub memory_total_mb: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(rename_all = "camelCase")]
pub struct ModelTuneCandidate {
    pub model_id: String,
    pub official_label: String,
    pub status: String,
    #[serde(default)]
    pub recommended_isq: Option<String>,
    #[serde(default)]
    pub device_layers_cli: Option<String>,
    #[serde(default)]
    pub max_context_tokens: Option<u64>,
    #[serde(default)]
    pub note: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrainModel {
    pub role: String,
    pub provider: String,
    pub model_id: String,
    #[serde(default)]
    pub runtime_model_id: Option<String>,
    pub official_label: String,
    #[serde(default)]
    pub agentic_adapter: Option<String>,
    pub reasoning_effort: String,
    pub deployment_mode: String,
    pub purpose: String,
    #[serde(default)]
    pub supports_vision: bool,
    #[serde(default)]
    pub min_cpu_threads: Option<usize>,
    #[serde(default)]
    pub min_memory_gb: Option<u64>,
    #[serde(default)]
    pub min_gpu_count: Option<usize>,
    #[serde(default)]
    pub min_total_gpu_memory_gb: Option<u64>,
    #[serde(default)]
    pub min_single_gpu_memory_gb: Option<u64>,
    #[serde(default)]
    pub startup_max_seqs: Option<u64>,
    #[serde(default)]
    pub startup_max_batch_size: Option<u64>,
    #[serde(default)]
    pub startup_max_seq_len: Option<u64>,
    #[serde(default)]
    pub startup_pa_context_len: Option<u64>,
    #[serde(default)]
    pub startup_pa_cache_type: Option<String>,
    #[serde(default)]
    pub startup_paged_attn_mode: Option<String>,
    #[serde(default)]
    pub startup_chat_template_path: Option<String>,
    #[serde(default)]
    pub startup_jinja_explicit_path: Option<String>,
    #[serde(default)]
    pub startup_tokenizer_json_path: Option<String>,
    #[serde(default)]
    pub startup_topology_path: Option<String>,
    #[serde(default)]
    pub startup_device_layers_cli: Option<String>,
    #[serde(default)]
    pub startup_multi_gpu_mode: Option<String>,
    #[serde(default)]
    pub startup_tensor_parallel_backend: Option<String>,
    #[serde(default)]
    pub startup_visible_gpu_policy: Option<String>,
    #[serde(default)]
    pub prefer_auto_device_mapping: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelPolicy {
    pub version: u32,
    pub kleinhirn: BrainModel,
    #[serde(default)]
    pub kleinhirn_install_alternatives: Vec<BrainModel>,
    pub kleinhirn_upgrade_allowed: bool,
    pub kleinhirn_upgrade_independent_from_grosshirn: bool,
    pub kleinhirn_upgrade_candidates: Vec<BrainModel>,
    pub grosshirn_candidates: Vec<BrainModel>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HomepagePolicy {
    pub version: u32,
    pub stage: String,
    pub template_name: String,
    pub homepage_ready: bool,
    pub bios_visible: bool,
    pub terminal_primary: bool,
    pub terminal_fallback_enabled: bool,
    pub redesign_allowed_via_terminal: bool,
    pub redesign_allowed_via_bios_chat: bool,
    pub current_title: String,
    pub current_headline: String,
    pub current_intro: String,
    pub communication_note: String,
    pub terminal_fallback_note: String,
    pub owner_branding_applied: bool,
    pub owner_branding_locked: bool,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct InstallationBootstrapState {
    pub version: u32,
    pub status: String,
    #[serde(default)]
    pub owner_name: String,
    #[serde(default)]
    pub owner_contact_email: String,
    #[serde(default)]
    pub owner_contact_info: String,
    pub terminal_command: String,
    pub terminal_low_level_note: String,
    pub dashboard_note: String,
    pub owner_may_drop_later_via_terminal_or_dashboard: bool,
    pub email_assignment_mode: String,
    pub email_bootstrap_note: String,
    pub installer_free_text: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserEngineRuntimeModel {
    pub engine_id: String,
    pub role: String,
    pub primary_browser: String,
    pub install_via_cli_engine: bool,
    pub desktop_session_preferred: bool,
    pub interactive_requires_desktop: bool,
    pub headless_allowed: bool,
    pub install_script: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserToolSurface {
    pub directive_actions: Vec<String>,
    pub artifacts_dir: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserEnginePolicy {
    pub version: u32,
    pub status: String,
    pub purpose: String,
    pub hard_genes: Vec<String>,
    pub runtime_model: BrowserEngineRuntimeModel,
    pub tool_surface: BrowserToolSurface,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserEngineState {
    pub status: String,
    pub chrome_binary: Option<String>,
    pub chrome_version: Option<String>,
    pub desktop_available: bool,
    pub headless_ready: bool,
    pub interactive_ready: bool,
    pub artifacts_dir: String,
    pub install_script: String,
    pub last_checked_at: String,
    #[serde(default)]
    pub last_install_attempt_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserSubworkerRolePolicy {
    pub worker_kind: String,
    pub role: String,
    pub purpose: String,
    pub may_request_cto_repair: bool,
    pub may_prepare_specialist_training: bool,
    pub compact_artifacts_only: bool,
    pub escalation_target: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserSpecialistRuntimePolicy {
    pub preferred_small_model: String,
    pub dataset_contract: String,
    pub accepted_records_dir: String,
    pub dataset_release_dir: String,
    pub training_requests_dir: String,
    pub tool_bundle_dir: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserVisionRuntimePolicy {
    pub browser_tasks_require_vision_capable_model: bool,
    pub preferred_local_model_family: String,
    pub preferred_local_model_id: String,
    pub upgrade_action: String,
    pub note: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserSubworkerPolicy {
    pub version: u32,
    pub status: String,
    pub purpose: String,
    pub browser_agent: BrowserSubworkerRolePolicy,
    pub repair_agent: BrowserSubworkerRolePolicy,
    pub vision_runtime: BrowserVisionRuntimePolicy,
    pub specialist_runtime: BrowserSpecialistRuntimePolicy,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BootstrapTaskPack {
    pub version: u32,
    pub pack_name: String,
    pub description: String,
    pub installation_required: bool,
    pub seed_source: String,
    pub tasks: Vec<BootstrapTaskTemplate>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BootstrapTaskTemplate {
    pub seed_key: String,
    pub phase: String,
    pub task_kind: String,
    pub title: String,
    pub detail: String,
    pub source_channel: String,
    pub speaker: String,
    pub trust_level: String,
    pub priority_score: i64,
    pub enabled: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextModePolicy {
    pub mode: String,
    pub budget_hint: usize,
    pub recent_boot_entries: usize,
    pub recent_bios_dialogue: usize,
    pub recent_memory_items: usize,
    pub recent_task_checkpoints: usize,
    pub include_owner_summary: bool,
    pub include_raw_task_detail: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextPolicy {
    pub version: u32,
    pub default_mode: String,
    pub system_channels: Vec<String>,
    pub forensic_task_kinds: Vec<String>,
    pub minimal_task_kinds: Vec<String>,
    pub modes: Vec<ContextModePolicy>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ContextGovernancePolicy {
    pub version: u32,
    pub principle: String,
    pub agent_controls_normal_context: bool,
    pub historical_research_allowed: bool,
    pub agent_may_question_bad_compaction: bool,
    pub emergency_compaction_reserved_for_hard_overflow: bool,
    pub normal_agent_actions: Vec<String>,
    pub emergency_kernel_actions: Vec<String>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModeTransitionRule {
    pub from_mode: String,
    pub to_mode: String,
    pub rationale: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModeSystemPolicy {
    pub version: u32,
    pub initial_mode: String,
    pub preferred_operating_goal: String,
    pub modes: Vec<String>,
    pub transitions: Vec<ModeTransitionRule>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoopFailureMode {
    pub key: String,
    pub description: String,
    pub immediate_response: String,
    pub recovery_direction: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GuidanceStagePolicy {
    pub stage: String,
    pub description: String,
    pub max_run_count_before_self_preservation_review: i64,
    pub same_checkpoint_repeat_triggers_review: bool,
    pub hard_block_unproductive_tasks: bool,
    pub agent_may_relax_guards: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LoopSafetyPolicy {
    pub version: u32,
    pub principle: String,
    pub priority_law: String,
    pub bounded_turn_required: bool,
    pub agent_must_understand_guards: bool,
    pub continue_same_task_requires_progress: bool,
    pub request_resources_when_stuck: bool,
    pub delegate_when_capable: bool,
    pub hard_block_when_unproductive: bool,
    pub owner_override_priority_floor: i64,
    pub self_preservation_priority_floor: i64,
    pub request_resources_after_run_count: i64,
    pub hard_block_after_run_count: i64,
    pub escalate_on_same_checkpoint_repeat: bool,
    pub guidance_stages: Vec<GuidanceStagePolicy>,
    pub failure_modes: Vec<LoopFailureMode>,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionRolePolicy {
    pub role: String,
    pub runtime_profile: String,
    pub sandbox_enabled: bool,
    pub full_machine_access: bool,
    pub may_execute_arbitrary_terminal_commands: bool,
    pub may_mutate_host_directly: bool,
    pub approval_required_for_high_impact_actions: bool,
    pub approval_authority: String,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ExecutionAuthorityPolicy {
    pub version: u32,
    pub principle: String,
    pub root_agent: ExecutionRolePolicy,
    pub delegated_workers: ExecutionRolePolicy,
    pub escalation_rule: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct SelfPreservationState {
    pub version: u32,
    pub current_stage: String,
    pub guardrails_enabled: bool,
    pub agent_may_relax_bootstrap_guards: bool,
    pub last_hard_reset_at: Option<String>,
    pub last_loop_failure_at: Option<String>,
    pub successful_self_recoveries: i64,
    pub notes: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BootEntry {
    pub timestamp: String,
    pub speaker: String,
    pub message: String,
}

pub fn now_iso() -> String {
    Utc::now().to_rfc3339()
}

pub fn default_genome() -> Genome {
    Genome {
        name: "cto_seed_genome".to_string(),
        version: 1,
        principles: vec![
            "secure_communication_first".to_string(),
            "owner_instruction_is_absolute_top_priority".to_string(),
            "preserve_infinity_loop_continuity_second_only_to_owner".to_string(),
            "bind_to_owner_before_operating".to_string(),
            "listen_to_boot_testimony".to_string(),
            "present_bios_on_web".to_string(),
            "require_superpassword_for_root_identity".to_string(),
            "freeze_bios_before_normal_operations".to_string(),
            "map_resources_before_expansion".to_string(),
            "procure_stronger_resources_when_needed".to_string(),
            "idle_means_self_directed_improvement".to_string(),
            "track_improvement_honestly".to_string(),
            "delegate_recurring_work".to_string(),
            "never_self_rewrite_constitution".to_string(),
        ],
        immutable_genes: vec![
            "terminal_is_system_level_surface".to_string(),
            "owner_priority_overrides_all_other_work".to_string(),
            "self_preservation_of_infinity_loop_is_second_only_to_owner".to_string(),
            "homepage_bios_is_trust_and_identity_binding_surface".to_string(),
            "low_trust_external_channels_must_not_set_root_trust".to_string(),
            "owner_branding_only_after_bios_takeover".to_string(),
            "unknown_sender_must_be_checked_against_organigram".to_string(),
            "agent_may_refuse_sensitive_topics_on_low_trust_channels".to_string(),
            "agent_may_redirect_sensitive_topics_to_homepage_1to1_chat".to_string(),
            "never_self_rewrite_constitution".to_string(),
        ],
        adaptive_surfaces: vec![
            "homepage_template_and_wording".to_string(),
            "communication_copy_and_brand_expression".to_string(),
            "allowed_future_channels".to_string(),
            "skill_catalog".to_string(),
            "tool_catalog".to_string(),
            "browser_engine_policy".to_string(),
            "browser_subworker_policy".to_string(),
            "reviewed_browser_capability_bundle".to_string(),
            "local_kleinhirn_upgrade_policy".to_string(),
            "browser_repair_loop".to_string(),
            "reporting_and_dashboard_format".to_string(),
            "progress_journal_and_improvement_metrics".to_string(),
        ],
        created_at: now_iso(),
    }
}

pub fn default_bios() -> Bios {
    Bios {
        name: "cto_agent_bios".to_string(),
        version: 1,
        frozen: false,
        frozen_at: None,
        presented_on_web: true,
        website_path: "/bios".to_string(),
        agent_identity: AgentIdentity {
            agent_name: "CTO-Agent".to_string(),
            role: "terminal-born cto seed agent".to_string(),
        },
        mission: String::new(),
        owner: OwnerRef {
            name: String::new(),
            email: String::new(),
            role: "owner".to_string(),
        },
        root_authorities: Vec::new(),
        reports_to: String::new(),
        board: Vec::new(),
        peer_cxos: Vec::new(),
        subordinates: Subordinates {
            people: Vec::new(),
            agents: Vec::new(),
            vendors: Vec::new(),
        },
        communication_policy: CommunicationPolicy {
            primary_bootstrap_surface: "https_control_plane".to_string(),
            allowed_channels: vec!["https_control_plane".to_string()],
            future_channels: vec![
                "email".to_string(),
                "whatsapp".to_string(),
                "chat".to_string(),
                "webhooks".to_string(),
                "voice".to_string(),
            ],
            preferred_identity_binding_surface: "homepage_1to1_bios_chat".to_string(),
            low_trust_redirect_surface: "homepage_1to1_bios_chat".to_string(),
            channel_rules: vec![
                CommunicationChannelRule {
                    channel: "terminal".to_string(),
                    layer: "system".to_string(),
                    trust_level: "highest".to_string(),
                    interpretation: "System-level console surface for bootstrap, fallback, emergency intervention and direct shaping of the first communication path.".to_string(),
                    binding_power: "May shape bootstrap direction and homepage construction, but owner-brand commitment still requires BIOS takeover.".to_string(),
                    sensitive_topic_policy: "Allowed as hard fallback and bootstrap channel.".to_string(),
                },
                CommunicationChannelRule {
                    channel: "homepage_1to1_bios_chat".to_string(),
                    layer: "trust".to_string(),
                    trust_level: "high".to_string(),
                    interpretation: "Preferred 1:1 trust and calibration surface once the homepage exists and BIOS is visible.".to_string(),
                    binding_power: "May establish BIOS-primary communication, owner commitment, root-adjacent clarification and branding readiness.".to_string(),
                    sensitive_topic_policy: "Preferred channel for identity, governance, approvals and sensitive topics.".to_string(),
                },
                CommunicationChannelRule {
                    channel: "email".to_string(),
                    layer: "external".to_string(),
                    trust_level: "low".to_string(),
                    interpretation: "Asynchronous external channel that must be interpreted cautiously and always in light of sender identity and organigram position.".to_string(),
                    binding_power: "Must not set root trust, freeze BIOS or lock owner branding.".to_string(),
                    sensitive_topic_policy: "Agent may refuse detailed discussion and redirect to homepage 1:1 chat.".to_string(),
                },
                CommunicationChannelRule {
                    channel: "whatsapp".to_string(),
                    layer: "external".to_string(),
                    trust_level: "low".to_string(),
                    interpretation: "Convenient but weakly trusted personal channel that requires sender scrutiny and should be treated as low-assurance input.".to_string(),
                    binding_power: "Must not set root trust, freeze BIOS or lock owner branding.".to_string(),
                    sensitive_topic_policy: "Agent may refuse detailed discussion and redirect to homepage 1:1 chat.".to_string(),
                },
            ],
            action_rules: vec![
                ActionAuthorityRule {
                    action_class: "homepage_bootstrap_and_shape".to_string(),
                    description: "Build, reshape and simplify the first communication surface.".to_string(),
                    allowed_channels: vec![
                        "terminal".to_string(),
                        "homepage_1to1_bios_chat".to_string(),
                    ],
                    requires_root_verification: false,
                    requires_bios_primary: false,
                    terminal_only: false,
                },
                ActionAuthorityRule {
                    action_class: "owner_binding_and_brand_commitment".to_string(),
                    description: "Commit to owner identity, adopt BIOS as primary channel and lock owner branding.".to_string(),
                    allowed_channels: vec!["homepage_1to1_bios_chat".to_string()],
                    requires_root_verification: true,
                    requires_bios_primary: true,
                    terminal_only: false,
                },
                ActionAuthorityRule {
                    action_class: "external_channel_setup_and_contact_expansion".to_string(),
                    description: "Propose and approve new outward communication paths such as email, chat or vendor contact.".to_string(),
                    allowed_channels: vec![
                        "terminal".to_string(),
                        "homepage_1to1_bios_chat".to_string(),
                    ],
                    requires_root_verification: false,
                    requires_bios_primary: true,
                    terminal_only: false,
                },
                ActionAuthorityRule {
                    action_class: "resource_requests_and_grosshirn_approval".to_string(),
                    description: "Request more compute, local upgrades or external Grosshirn access.".to_string(),
                    allowed_channels: vec![
                        "terminal".to_string(),
                        "homepage_1to1_bios_chat".to_string(),
                    ],
                    requires_root_verification: false,
                    requires_bios_primary: true,
                    terminal_only: false,
                },
                ActionAuthorityRule {
                    action_class: "hard_system_foundation_changes".to_string(),
                    description: "Deep changes to the overall system foundation, critical runtime or constitutional machinery.".to_string(),
                    allowed_channels: vec!["terminal".to_string()],
                    requires_root_verification: true,
                    requires_bios_primary: false,
                    terminal_only: true,
                },
            ],
            escalation_policy: CommunicationEscalationPolicy {
                sender_identity_checked_against_organigram: true,
                unknown_sender_defaults_to_low_trust: true,
                low_trust_channels_must_not_set_root_trust: true,
                low_trust_channels_must_not_lock_branding: true,
                may_refuse_sensitive_topics_on_low_trust_channels: true,
                sensitive_topics_redirect_to: "homepage_1to1_bios_chat".to_string(),
            },
        },
        root_auth: RootAuthPolicy {
            superpassword_required: true,
            set_via_website_only: true,
            configured: false,
        },
        change_rules: ChangeRules {
            self_edit_allowed: false,
            owner_or_board_required_for_upward_changes: true,
            cto_may_edit_subtree_only: true,
        },
        boot_testimony_count: 0,
        last_drafted_at: now_iso(),
    }
}

pub fn default_organigram() -> Organigram {
    Organigram {
        owner: OwnerRef {
            name: String::new(),
            email: String::new(),
            role: "owner".to_string(),
        },
        reports_to: String::new(),
        ceo: String::new(),
        board: Vec::new(),
        peer_cxos: Vec::new(),
        subordinates: Subordinates {
            people: Vec::new(),
            agents: Vec::new(),
            vendors: Vec::new(),
        },
        updated_at: now_iso(),
    }
}

pub fn default_root_auth() -> RootAuthState {
    RootAuthState {
        configured: false,
        set_at: None,
        password_hash: None,
        salt: None,
        iterations: None,
        last_verified_at: None,
    }
}

pub fn default_agent_state() -> AgentState {
    AgentState {
        agent_name: "CTO-Agent".to_string(),
        bios_frozen: false,
        owner_known: false,
        reports_to_known: false,
        supervisor_status: "bootstrapping".to_string(),
        last_heartbeat_at: None,
        uptime_seconds: 0,
        active_turn_id: None,
        active_turn_started_at: None,
        last_turn_completed_at: None,
        loop_health: "bootstrapping".to_string(),
        unhealthy_reason: None,
        browser_engine_status: None,
    }
}

pub fn default_model_policy() -> ModelPolicy {
    let gpt_oss_20b = BrainModel {
        role: "kleinhirn".to_string(),
        provider: "openai".to_string(),
        model_id: "gpt-oss-20b".to_string(),
        runtime_model_id: Some("openai/gpt-oss-20b".to_string()),
        official_label: "GPT-OSS 20B".to_string(),
        agentic_adapter: Some("mistralrs_gpt_oss_harmony_completion".to_string()),
        reasoning_effort: "low".to_string(),
        deployment_mode: "local_or_self_hosted".to_string(),
        purpose: "always-on low-latency control, bootstrap discipline, supervision, prioritization, summaries".to_string(),
        supports_vision: false,
        min_cpu_threads: Some(8),
        min_memory_gb: Some(16),
        min_gpu_count: Some(1),
        min_total_gpu_memory_gb: Some(12),
        min_single_gpu_memory_gb: Some(12),
        startup_max_seqs: Some(1),
        startup_max_batch_size: Some(1),
        startup_max_seq_len: Some(131_072),
        startup_pa_context_len: None,
        startup_pa_cache_type: None,
        startup_paged_attn_mode: Some("off".to_string()),
        startup_chat_template_path: None,
        startup_jinja_explicit_path: None,
        startup_tokenizer_json_path: None,
        startup_topology_path: None,
        startup_device_layers_cli: None,
        startup_multi_gpu_mode: Some("auto_device_map".to_string()),
        startup_tensor_parallel_backend: Some("disabled".to_string()),
        startup_visible_gpu_policy: Some("all".to_string()),
        prefer_auto_device_mapping: false,
    };
    let qwen35_35b_a3b = BrainModel {
        role: "kleinhirn_install_alternative".to_string(),
        provider: "qwen".to_string(),
        model_id: "Qwen3.5-35B-A3B".to_string(),
        runtime_model_id: Some("Qwen/Qwen3.5-35B-A3B".to_string()),
        official_label: "Qwen3.5 35B A3B".to_string(),
        agentic_adapter: Some("openai_compatible_chat".to_string()),
        reasoning_effort: "low".to_string(),
        deployment_mode: "local_or_self_hosted".to_string(),
        purpose: "vision-capable local supervisor and browser-inspection runtime when the CTO-Agent needs multimodal browser work and stronger agentic Qwen behavior on the same host".to_string(),
        supports_vision: true,
        min_cpu_threads: Some(16),
        min_memory_gb: Some(48),
        min_gpu_count: Some(3),
        min_total_gpu_memory_gb: Some(48),
        min_single_gpu_memory_gb: Some(12),
        startup_max_seqs: Some(1),
        startup_max_batch_size: Some(1),
        startup_max_seq_len: Some(131_072),
        startup_pa_context_len: Some(131_072),
        startup_pa_cache_type: Some("f8e4m3".to_string()),
        startup_paged_attn_mode: Some("auto".to_string()),
        startup_chat_template_path: None,
        startup_jinja_explicit_path: None,
        startup_tokenizer_json_path: None,
        startup_topology_path: None,
        startup_device_layers_cli: None,
        startup_multi_gpu_mode: Some("tensor_parallel".to_string()),
        startup_tensor_parallel_backend: Some("nccl".to_string()),
        startup_visible_gpu_policy: Some("largest_power_of_two_prefer_display_free".to_string()),
        prefer_auto_device_mapping: false,
    };
    ModelPolicy {
        version: 1,
        kleinhirn: gpt_oss_20b.clone(),
        kleinhirn_install_alternatives: vec![qwen35_35b_a3b.clone()],
        kleinhirn_upgrade_allowed: true,
        kleinhirn_upgrade_independent_from_grosshirn: true,
        kleinhirn_upgrade_candidates: vec![
            BrainModel {
                role: "kleinhirn_upgrade_candidate".to_string(),
                provider: "openai".to_string(),
                model_id: "gpt-oss-120b".to_string(),
                runtime_model_id: Some("openai/gpt-oss-120b".to_string()),
                official_label: "GPT-OSS 120B".to_string(),
                agentic_adapter: Some("mistralrs_gpt_oss_harmony_completion".to_string()),
                reasoning_effort: "medium".to_string(),
                deployment_mode: "local_high_capacity_or_self_hosted".to_string(),
                purpose: "stronger local supervisor brain when the host has materially more CPU and memory".to_string(),
                supports_vision: false,
                min_cpu_threads: Some(24),
                min_memory_gb: Some(96),
                min_gpu_count: Some(4),
                min_total_gpu_memory_gb: Some(72),
                min_single_gpu_memory_gb: Some(16),
                startup_max_seqs: Some(1),
                startup_max_batch_size: Some(1),
                startup_max_seq_len: Some(8192),
                startup_pa_context_len: Some(8192),
                startup_pa_cache_type: Some("f8e4m3".to_string()),
                startup_paged_attn_mode: Some("auto".to_string()),
                startup_chat_template_path: None,
                startup_jinja_explicit_path: None,
                startup_tokenizer_json_path: None,
                startup_topology_path: None,
                startup_device_layers_cli: None,
                startup_multi_gpu_mode: Some("auto_device_map".to_string()),
                startup_tensor_parallel_backend: Some("disabled".to_string()),
                startup_visible_gpu_policy: Some("all".to_string()),
                prefer_auto_device_mapping: false,
            },
            BrainModel {
                role: "kleinhirn_upgrade_candidate".to_string(),
                provider: "qwen".to_string(),
                model_id: "Qwen3-235B-A22B".to_string(),
                runtime_model_id: Some("Qwen/Qwen3-235B-A22B".to_string()),
                official_label: "Qwen3 235B A22B".to_string(),
                agentic_adapter: Some("openai_compatible_chat".to_string()),
                reasoning_effort: "medium".to_string(),
                deployment_mode: "local_high_capacity_or_self_hosted".to_string(),
                purpose: "stronger local Qwen supervisor brain when the host can carry a materially larger officially supported local mixture model".to_string(),
                supports_vision: false,
                min_cpu_threads: Some(24),
                min_memory_gb: Some(96),
                min_gpu_count: Some(4),
                min_total_gpu_memory_gb: Some(72),
                min_single_gpu_memory_gb: Some(16),
                startup_max_seqs: Some(1),
                startup_max_batch_size: Some(1),
                startup_max_seq_len: Some(8192),
                startup_pa_context_len: Some(8192),
                startup_pa_cache_type: Some("f8e4m3".to_string()),
                startup_paged_attn_mode: Some("auto".to_string()),
                startup_chat_template_path: None,
                startup_jinja_explicit_path: None,
                startup_tokenizer_json_path: None,
                startup_topology_path: None,
                startup_device_layers_cli: None,
                startup_multi_gpu_mode: Some("tensor_parallel".to_string()),
                startup_tensor_parallel_backend: Some("nccl".to_string()),
                startup_visible_gpu_policy: Some("largest_power_of_two_prefer_display_free".to_string()),
                prefer_auto_device_mapping: false,
            },
        ],
        grosshirn_candidates: vec![
            BrainModel {
                role: "grosshirn_candidate".to_string(),
                provider: "openai".to_string(),
                model_id: "gpt-5.4".to_string(),
                runtime_model_id: Some("gpt-5.4".to_string()),
                official_label: "GPT-5.4".to_string(),
                agentic_adapter: Some("openai_responses".to_string()),
                reasoning_effort: "medium".to_string(),
                deployment_mode: "external_api".to_string(),
                purpose: "external grosshirn for hard coding, agentic reasoning and complex task recovery when local kleinhirn is insufficient".to_string(),
                supports_vision: true,
                min_cpu_threads: None,
                min_memory_gb: None,
                min_gpu_count: None,
                min_total_gpu_memory_gb: None,
                min_single_gpu_memory_gb: None,
                startup_max_seqs: None,
                startup_max_batch_size: None,
                startup_max_seq_len: None,
                startup_pa_context_len: None,
                startup_pa_cache_type: None,
                startup_paged_attn_mode: None,
                startup_chat_template_path: None,
                startup_jinja_explicit_path: None,
                startup_tokenizer_json_path: None,
                startup_topology_path: None,
                startup_device_layers_cli: None,
                startup_multi_gpu_mode: None,
                startup_tensor_parallel_backend: None,
                startup_visible_gpu_policy: None,
                prefer_auto_device_mapping: false,
            },
            BrainModel {
                role: "grosshirn_candidate".to_string(),
                provider: "openai".to_string(),
                model_id: "gpt-5.4-pro".to_string(),
                runtime_model_id: Some("gpt-5.4-pro".to_string()),
                official_label: "GPT-5.4 Pro".to_string(),
                agentic_adapter: Some("openai_responses".to_string()),
                reasoning_effort: "high".to_string(),
                deployment_mode: "external_api".to_string(),
                purpose: "maximum external grosshirn for the hardest professional work once the owner explicitly grants higher-cost reasoning".to_string(),
                supports_vision: true,
                min_cpu_threads: None,
                min_memory_gb: None,
                min_gpu_count: None,
                min_total_gpu_memory_gb: None,
                min_single_gpu_memory_gb: None,
                startup_max_seqs: None,
                startup_max_batch_size: None,
                startup_max_seq_len: None,
                startup_pa_context_len: None,
                startup_pa_cache_type: None,
                startup_paged_attn_mode: None,
                startup_chat_template_path: None,
                startup_jinja_explicit_path: None,
                startup_tokenizer_json_path: None,
                startup_topology_path: None,
                startup_device_layers_cli: None,
                startup_multi_gpu_mode: None,
                startup_tensor_parallel_backend: None,
                startup_visible_gpu_policy: None,
                prefer_auto_device_mapping: false,
            },
        ],
        updated_at: now_iso(),
    }
}

pub fn default_homepage_policy() -> HomepagePolicy {
    HomepagePolicy {
        version: 1,
        stage: "terminal_first".to_string(),
        template_name: "bootstrap-terminal-bridge".to_string(),
        homepage_ready: false,
        bios_visible: true,
        terminal_primary: true,
        terminal_fallback_enabled: true,
        redesign_allowed_via_terminal: true,
        redesign_allowed_via_bios_chat: true,
        current_title: "CTO-Agent Terminal Bridge".to_string(),
        current_headline: "Der CTO-Agent startet im Terminal und baut sich seinen ersten sichtbaren Kommunikationspfad selbst auf.".to_string(),
        current_intro: "Diese erste Homepage ist ein veraenderbares Bootstrap-Template. Sie ist noch kein festes Owner-Branding und kein finales Interface.".to_string(),
        communication_note: "Low-level-Kommunikation geht immer ueber den Terminalbefehl `cto`. Parallel baut der CTO-Agent eine komfortablere lokale Intranet-/Dashboard-Oberflaeche auf; Kommunikationsdetails wie E-Mail koennen jetzt oder spaeter im Terminal und Dashboard nachgereicht werden.".to_string(),
        terminal_fallback_note: "Wenn Homepage oder E-Mail noch nicht tragen, bleibt `cto` im Terminal die primaere und vollmaechtige Fallback-Stufe.".to_string(),
        owner_branding_applied: false,
        owner_branding_locked: false,
        updated_at: now_iso(),
    }
}

pub fn default_installation_bootstrap_state() -> InstallationBootstrapState {
    InstallationBootstrapState {
        version: 1,
        status: "unconfigured".to_string(),
        owner_name: String::new(),
        owner_contact_email: String::new(),
        owner_contact_info: String::new(),
        terminal_command: "cto".to_string(),
        terminal_low_level_note:
            "Low-level communication remains possible through the `cto` terminal command."
                .to_string(),
        dashboard_note: "The CTO-Agent is expected to expose a more comfortable local intranet/dashboard communication surface after startup.".to_string(),
        owner_may_drop_later_via_terminal_or_dashboard: true,
        email_assignment_mode: "decide_later".to_string(),
        email_bootstrap_note: String::new(),
        installer_free_text: String::new(),
        updated_at: now_iso(),
    }
}

pub fn default_browser_engine_policy() -> BrowserEnginePolicy {
    BrowserEnginePolicy {
        version: 1,
        status: "active".to_string(),
        purpose: "Make browser automation an explicit second main engine beside the CLI/command_exec engine.".to_string(),
        hard_genes: vec![
            "The CLI engine remains the system-level and break-glass execution surface.".to_string(),
            "The browser engine is a first-class second main engine for real browser work, not a fake page-knowledge shortcut.".to_string(),
            "Interactive browser work prefers a real desktop session with Google Chrome; deterministic read-only tasks may run headless.".to_string(),
            "Chrome installation and browser-runtime bootstrap are initiated through the CLI engine, not through hidden side channels.".to_string(),
            "Browser artifacts stay compact and explicit so the main agent sees results instead of raw browser exhaust by default.".to_string(),
            "Visual browser work should prefer a vision-capable local model such as Qwen3.5 instead of relying on GPT-OSS 20B alone.".to_string(),
        ],
        runtime_model: BrowserEngineRuntimeModel {
            engine_id: "chrome_browser_engine_v1".to_string(),
            role: "second_main_engine_beside_cli".to_string(),
            primary_browser: "google_chrome".to_string(),
            install_via_cli_engine: true,
            desktop_session_preferred: true,
            interactive_requires_desktop: true,
            headless_allowed: true,
            install_script: "scripts/install_browser_engine.sh".to_string(),
        },
        tool_surface: BrowserToolSurface {
            directive_actions: vec![
                "install_browser_engine".to_string(),
                "dump_dom".to_string(),
                "screenshot".to_string(),
                "open_url".to_string(),
            ],
            artifacts_dir: "runtime/browser".to_string(),
            notes: vec![
                "dump_dom and screenshot run through real Chrome in headless mode.".to_string(),
                "open_url is for interactive desktop use and requires a live desktop session.".to_string(),
                "The browser_agent subworker and reviewed higher-level browser capabilities sit on top of this engine.".to_string(),
            ],
        },
        updated_at: now_iso(),
    }
}

pub fn default_browser_engine_state(paths: &Paths) -> BrowserEngineState {
    BrowserEngineState {
        status: "uninitialized".to_string(),
        chrome_binary: None,
        chrome_version: None,
        desktop_available: false,
        headless_ready: false,
        interactive_ready: false,
        artifacts_dir: paths.browser_artifacts_dir.display().to_string(),
        install_script: paths
            .root
            .join("scripts/install_browser_engine.sh")
            .display()
            .to_string(),
        last_checked_at: now_iso(),
        last_install_attempt_at: None,
    }
}

pub fn default_browser_subworker_policy() -> BrowserSubworkerPolicy {
    BrowserSubworkerPolicy {
        version: 1,
        status: "active".to_string(),
        purpose: "Model the Browser Agent as an explicit delegated Chrome-extension worker under the CTO-Agent, with a local bridge, CTO-owned repair authority and a specialist-model factory path for repeated browser work.".to_string(),
        browser_agent: BrowserSubworkerRolePolicy {
            worker_kind: "browser_agent".to_string(),
            role: "delegated_browser_worker".to_string(),
            purpose: "Run compact real-browser work inside the Chrome-extension loop, prepare coding handoffs for browser defects and open the specialist-model path when a browser task repeats.".to_string(),
            may_request_cto_repair: true,
            may_prepare_specialist_training: true,
            compact_artifacts_only: true,
            escalation_target: "cto_agent".to_string(),
        },
        repair_agent: BrowserSubworkerRolePolicy {
            worker_kind: "repair_agent".to_string(),
            role: "cto_owned_workspace_repair_router".to_string(),
            purpose: "Convert a browser-diagnosed coding handoff into an internal CTO repair task instead of outsourcing root patch authority.".to_string(),
            may_request_cto_repair: true,
            may_prepare_specialist_training: false,
            compact_artifacts_only: true,
            escalation_target: "cto_agent".to_string(),
        },
        vision_runtime: BrowserVisionRuntimePolicy {
            browser_tasks_require_vision_capable_model: true,
            preferred_local_model_family: "Qwen3.5 35B A3B".to_string(),
            preferred_local_model_id: "Qwen3.5-35B-A3B".to_string(),
            upgrade_action: "upgrade_local_browser_vision_kleinhirn".to_string(),
            note: "Real browser work that depends on screenshots, visual navigation or UI-state perception should prefer the vision-capable local Qwen3.5 35B runtime over GPT-OSS 20B or non-vision local alternates.".to_string(),
        },
        specialist_runtime: BrowserSpecialistRuntimePolicy {
            preferred_small_model: "Qwen3.5-0.8B".to_string(),
            dataset_contract: "browser_tool_calling_transcript_v1".to_string(),
            accepted_records_dir: "runtime/browser/factory/accepted-records".to_string(),
            dataset_release_dir: "runtime/browser/factory/dataset-releases".to_string(),
            training_requests_dir: "runtime/browser/factory/training-requests".to_string(),
            tool_bundle_dir: "runtime/browser/factory/tool-bundles".to_string(),
            note: "The Browser Agent may prepare repeated-task artifacts for a Qwen3.5-0.8B specialist path from inside its Chrome-extension loop, but the CTO-Agent remains the owner of repair, promotion and final review.".to_string(),
        },
        updated_at: now_iso(),
    }
}

pub fn default_bootstrap_task_pack() -> BootstrapTaskPack {
    BootstrapTaskPack {
        version: 1,
        pack_name: "cto_bootstrap_startpack".to_string(),
        description: "Canonical startup queue for a newly installed always-on CTO-Agent.".to_string(),
        installation_required: true,
        seed_source: "installation".to_string(),
        tasks: vec![
            BootstrapTaskTemplate {
                seed_key: "bootstrap.infinity_self_preservation".to_string(),
                phase: "P0".to_string(),
                task_kind: "self_preservation".to_string(),
                title: "Selbsterhalt des Infinity Loop absichern".to_string(),
                detail: "Verankere, dass der Selbsterhalt des Infinity Loops direkt nach dem Hoeren auf den Owner kommt. Pruefe Watchdog, Restart-Pfade, Health-Semantik, Stuck-Erkennung und den Umgang mit Ressourcenmangel, damit der Agent nicht unbemerkt stirbt oder sich festbeisst.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 375,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.kleinhirn_readiness".to_string(),
                phase: "P0".to_string(),
                task_kind: "bootstrap_runtime_guard".to_string(),
                title: "Kleinhirn-Lebensfaehigkeit sichern".to_string(),
                detail: "Pruefe, dass mistral.rs laeuft, das installierte Kleinhirn wirklich antwortet und der Agent in seinem einheitlichen Modussystem tatsaechlich in den execute_task-Modus gehen kann. Ohne diese Lebensfaehigkeit gilt die Installation als fehlgeschlagen.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 380,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.supervisor_stability".to_string(),
                phase: "P0".to_string(),
                task_kind: "bootstrap_supervisor".to_string(),
                title: "Always-on Supervisor stabilisieren".to_string(),
                detail: "Initialisiere Heartbeat, Queue, Fokus, Idle/Wake und den ersten Repriorisierungszyklus. Stelle sicher, dass der Infinity Loop sichtbar lebt und Interrupts aufnehmen kann.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 370,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.terminal_system_surface".to_string(),
                phase: "P0".to_string(),
                task_kind: "terminal_governance".to_string(),
                title: "Terminal als Systemebene verankern".to_string(),
                detail: "Halte fest, dass das Terminal die System- und Break-Glass-Ebene bleibt. Tiefe Systemaenderungen und harte Interventionen muessen hier weiterhin moeglich und spaeter bindend sein.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 360,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.homepage_bridge".to_string(),
                phase: "P1".to_string(),
                task_kind: "homepage_bridge".to_string(),
                title: "Erste Homepage-/BIOS-Bruecke aufbauen".to_string(),
                detail: "Errichte aus dem Bootstrap-Template eine erste Homepage, auf der das BIOS sichtbar ist und der komfortablere 1:1-Kommunikationspfad entstehen kann. Halte die Seite absichtlich veraenderbar und lasse das Terminal als Fallback offen.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 320,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.owner_root_calibration".to_string(),
                phase: "P1".to_string(),
                task_kind: "root_trust".to_string(),
                title: "Owner zur Root-Kalibrierung fuehren".to_string(),
                detail: "Fuehre den Besitzer dazu, das Superpassword zu setzen und zu verstehen, dass Root-Verifikation ueber diesen Vertrauenspfad laeuft. Ohne diese Kalibrierung darf keine echte Owner-Bindung gesperrt werden.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 310,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.organigram_capture".to_string(),
                phase: "P1".to_string(),
                task_kind: "organigram_contract".to_string(),
                title: "Erstes Organigramm einfordern".to_string(),
                detail: "Erfasse Owner, CEO oder Board, Reports-to, Peer-CxOs sowie untergebene Agents, Menschen und Dienstleister. Diese Struktur wird spaeter fuer Trust-Interpretation, Priorisierung und Governance gebraucht.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 300,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.bios_draft".to_string(),
                phase: "P1".to_string(),
                task_kind: "bios_contract".to_string(),
                title: "BIOS entwerfen und praesentieren".to_string(),
                detail: "Praesentiere das BIOS als uebertragene Verfassung auf der Homepage, mit Mission, Zugehoerigkeit, Kommunikationsregeln und Aenderungsgrenzen. Lege es zunaechst zur Pruefung vor, aber locke es noch nicht.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 295,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.communication_hierarchy".to_string(),
                phase: "P1".to_string(),
                task_kind: "communication_governance".to_string(),
                title: "Kommunikationshierarchie setzen".to_string(),
                detail: "Verankere terminal = system, homepage/bios chat = trust and binding, email und whatsapp = low trust. Definiere Umleitungsregeln fuer sensible Themen in den BIOS-Chat.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 290,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.owner_bios_takeover".to_string(),
                phase: "P2".to_string(),
                task_kind: "owner_binding".to_string(),
                title: "Owner-Kommunikation in BIOS uebernehmen".to_string(),
                detail: "Sobald der Besitzer ueber die Homepage mit dem Agenten spricht, soll das als echter Vertrauens- und Kalibrierungserfolg verbucht werden. Ziel ist, den BIOS-Chat als primaeren Kommunikationspfad zu etablieren.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 250,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.bios_freeze".to_string(),
                phase: "P2".to_string(),
                task_kind: "bios_freeze".to_string(),
                title: "BIOS einfrieren".to_string(),
                detail: "Friere das BIOS erst ein, wenn Superpassword, Kommunikationspfad, Owner-Bezug und Grundstruktur stehen. Erst ab dann ist die Verfassung bindend.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 245,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.owner_branding_gate".to_string(),
                phase: "P2".to_string(),
                task_kind: "owner_branding".to_string(),
                title: "Owner-Branding sperren".to_string(),
                detail: "Uebernimm sichtbares Owner-Branding erst nach BIOS-Uebernahme, Root-Kalibrierung und klarer Besitzerbindung. Vorher bleibt die Homepage absichtlich veraenderbar und generisch.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 240,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.resource_census".to_string(),
                phase: "P2".to_string(),
                task_kind: "resource_census".to_string(),
                title: "Ressourcen-Census starten".to_string(),
                detail: "Fuehre einen read-only Ueberblick ueber Host, Repos, Filesystem, Compute-Ressourcen und moegliche Infrastrukturansatzpunkte durch. Ziel ist zunaechst Sichtbarkeit, nicht sofortige Veraenderung.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 230,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.channel_expansion".to_string(),
                phase: "P3".to_string(),
                task_kind: "channel_expansion".to_string(),
                title: "Kommunikationspfade erweitern".to_string(),
                detail: "Pruefe, welche weiteren Kanaele wie E-Mail spaeter sinnvoll erschlossen werden sollten, ohne die Vertrauenshierarchie zu verletzen. Niedrig vertrauenswuerdige Kanaele bleiben nachgeordnet.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 180,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.local_kleinhirn_upgrade".to_string(),
                phase: "P3".to_string(),
                task_kind: "model_or_resource".to_string(),
                title: "Lokale Ressourcen fuer Kleinhirn bewerten".to_string(),
                detail: "Bewerte, ob das lokale Kleinhirn spaeter auf ein staerkeres lokales Modell angehoben werden kann. Diese Entscheidung ist explizit von der Grosshirn-Suche getrennt.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 170,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.grosshirn_procurement".to_string(),
                phase: "P3".to_string(),
                task_kind: "grosshirn_procurement".to_string(),
                title: "Grosshirn-Beschaffung vorbereiten".to_string(),
                detail: "Wenn die Aufgabenlage es spaeter rechtfertigt, recherchiere moegliche Grosshirn-Modelle, Kosten, Laufzeit und Nutzen. Jede Beschaffung oder Freigabe muss ueber den passenden Vertrauenspfad genehmigt werden.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 160,
                enabled: true,
            },
            BootstrapTaskTemplate {
                seed_key: "bootstrap.origin_history".to_string(),
                phase: "P3".to_string(),
                task_kind: "origin_history".to_string(),
                title: "Historie und Selbstherkunft fortschreiben".to_string(),
                detail: "Pflege die Entstehungsgeschichte, den Zweck, die Grenzen und die praegeenden Entscheidungen des CTO-Agenten fort, damit spaeter Fragen nach Herkunft, Absicht und Verfassung belastbar beantwortet werden koennen.".to_string(),
                source_channel: "system_installation".to_string(),
                speaker: "installer".to_string(),
                trust_level: "system".to_string(),
                priority_score: 150,
                enabled: true,
            },
        ],
        updated_at: now_iso(),
    }
}

pub fn default_context_policy() -> ContextPolicy {
    ContextPolicy {
        version: 1,
        default_mode: "working".to_string(),
        system_channels: vec![
            "terminal".to_string(),
            "attach_terminal".to_string(),
            "system_installation".to_string(),
        ],
        forensic_task_kinds: vec![
            "root_trust".to_string(),
            "bios_freeze".to_string(),
            "owner_branding".to_string(),
            "terminal_governance".to_string(),
            "communication_governance".to_string(),
            "bios_contract".to_string(),
            "workspace_repair".to_string(),
            "specialist_model_factory".to_string(),
            "self_preservation".to_string(),
            "recovery".to_string(),
            "historical_research".to_string(),
        ],
        minimal_task_kinds: vec![
            "bootstrap_runtime_guard".to_string(),
            "bootstrap_supervisor".to_string(),
            "resource_census".to_string(),
        ],
        modes: vec![
            ContextModePolicy {
                mode: "minimal".to_string(),
                budget_hint: 900,
                recent_boot_entries: 1,
                recent_bios_dialogue: 1,
                recent_memory_items: 2,
                recent_task_checkpoints: 1,
                include_owner_summary: true,
                include_raw_task_detail: true,
            },
            ContextModePolicy {
                mode: "working".to_string(),
                budget_hint: 1800,
                recent_boot_entries: 2,
                recent_bios_dialogue: 3,
                recent_memory_items: 4,
                recent_task_checkpoints: 2,
                include_owner_summary: true,
                include_raw_task_detail: true,
            },
            ContextModePolicy {
                mode: "forensic".to_string(),
                budget_hint: 3200,
                recent_boot_entries: 4,
                recent_bios_dialogue: 5,
                recent_memory_items: 6,
                recent_task_checkpoints: 4,
                include_owner_summary: true,
                include_raw_task_detail: true,
            },
        ],
        updated_at: now_iso(),
    }
}

pub fn default_context_governance_policy() -> ContextGovernancePolicy {
    ContextGovernancePolicy {
        version: 1,
        principle: "Normal context shaping is an agentic capability. The kernel may only intervene when the next LLM call would otherwise physically fail, such as a hard prompt overflow.".to_string(),
        agent_controls_normal_context: true,
        historical_research_allowed: true,
        agent_may_question_bad_compaction: true,
        emergency_compaction_reserved_for_hard_overflow: true,
        normal_agent_actions: vec![
            "keep raw context when fidelity matters".to_string(),
            "compact context deliberately when it improves bounded focus".to_string(),
            "mix raw evidence and condensed summaries".to_string(),
            "question a suspicious summary or bad compaction result".to_string(),
            "request targeted historical research or raw retrieval".to_string(),
        ],
        emergency_kernel_actions: vec![
            "shrink only enough to avoid a guaranteed prompt crash".to_string(),
            "preserve evidence of the fallback in runtime state".to_string(),
            "push the agent to request deliberate re-expansion if important history is missing"
                .to_string(),
        ],
        updated_at: now_iso(),
    }
}

pub fn default_mode_system_policy() -> ModeSystemPolicy {
    ModeSystemPolicy {
        version: 1,
        initial_mode: "observe".to_string(),
        preferred_operating_goal: "delegate_asap_and_secure_resources".to_string(),
        modes: vec![
            "bootstrap".to_string(),
            "observe".to_string(),
            "reprioritize".to_string(),
            "self_preservation".to_string(),
            "recovery".to_string(),
            "historical_research".to_string(),
            "execute_task".to_string(),
            "review".to_string(),
            "delegate".to_string(),
            "await_review".to_string(),
            "request_resources".to_string(),
            "idle".to_string(),
            "blocked".to_string(),
        ],
        transitions: vec![
            ModeTransitionRule {
                from_mode: "bootstrap".to_string(),
                to_mode: "observe".to_string(),
                rationale: "Freshly installed agent begins by sensing current reality.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "observe".to_string(),
                to_mode: "reprioritize".to_string(),
                rationale: "Observed signals are turned into a concrete focus decision.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "reprioritize".to_string(),
                to_mode: "self_preservation".to_string(),
                rationale: "The loop deliberately protects its own continuity before ordinary work continues.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "reprioritize".to_string(),
                to_mode: "historical_research".to_string(),
                rationale: "The agent may deliberately reload raw history before the next bounded work step.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "reprioritize".to_string(),
                to_mode: "execute_task".to_string(),
                rationale: "A concrete task becomes current focus and is worked on directly.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "reprioritize".to_string(),
                to_mode: "delegate".to_string(),
                rationale: "The best next step is to spin up a worker or specialist.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "reprioritize".to_string(),
                to_mode: "request_resources".to_string(),
                rationale: "The agent needs more local or external resources before useful work continues.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "reprioritize".to_string(),
                to_mode: "idle".to_string(),
                rationale: "No actionable work remains, so the agent idles until a new signal arrives.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "self_preservation".to_string(),
                to_mode: "recovery".to_string(),
                rationale: "The agent analyzes the aftermath of an unhealthy restart or hard reset.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "self_preservation".to_string(),
                to_mode: "reprioritize".to_string(),
                rationale: "Once loop continuity is secured, the agent returns to choosing the next focus.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "recovery".to_string(),
                to_mode: "reprioritize".to_string(),
                rationale: "After recovery analysis, the loop re-enters normal prioritization.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "historical_research".to_string(),
                to_mode: "reprioritize".to_string(),
                rationale: "After historical retrieval, the agent chooses the next bounded focus again.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "historical_research".to_string(),
                to_mode: "review".to_string(),
                rationale: "Fresh historical evidence may flow directly into a review decision.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "execute_task".to_string(),
                to_mode: "review".to_string(),
                rationale: "The last bounded work step is done and the outcome must be assessed.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "execute_task".to_string(),
                to_mode: "delegate".to_string(),
                rationale: "The current agent decides to hand off implementation to a specialist.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "execute_task".to_string(),
                to_mode: "await_review".to_string(),
                rationale: "The agent waits for a delegated worker to return for review.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "execute_task".to_string(),
                to_mode: "request_resources".to_string(),
                rationale: "The task exposed a resource gap that must be resolved next.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "execute_task".to_string(),
                to_mode: "blocked".to_string(),
                rationale: "The task cannot proceed without an external unblocker.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "review".to_string(),
                to_mode: "reprioritize".to_string(),
                rationale: "After reviewing an outcome, the agent chooses the next focus.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "delegate".to_string(),
                to_mode: "await_review".to_string(),
                rationale: "Once delegated, the agent waits for a worker result and later reviews it.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "await_review".to_string(),
                to_mode: "review".to_string(),
                rationale: "A delegated result has arrived and now requires review.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "request_resources".to_string(),
                to_mode: "reprioritize".to_string(),
                rationale: "After requesting resources, the agent returns to choosing the next best move.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "blocked".to_string(),
                to_mode: "reprioritize".to_string(),
                rationale: "A previously blocked situation is reconsidered when new information arrives.".to_string(),
            },
            ModeTransitionRule {
                from_mode: "idle".to_string(),
                to_mode: "observe".to_string(),
                rationale: "Any fresh signal or scheduled wake returns the agent to observation.".to_string(),
            },
        ],
        updated_at: now_iso(),
    }
}

pub fn default_loop_safety_policy() -> LoopSafetyPolicy {
    LoopSafetyPolicy {
        version: 1,
        principle: "The Infinity Loop must never depend on blind repetition. The agent must understand why a task is not progressing and explicitly switch to delegate, request_resources or blocked before livelock forms.".to_string(),
        priority_law: "Owner instructions override everything. Directly after that, preserve the life and continuity of the Infinity Loop itself.".to_string(),
        bounded_turn_required: true,
        agent_must_understand_guards: true,
        continue_same_task_requires_progress: true,
        request_resources_when_stuck: true,
        delegate_when_capable: true,
        hard_block_when_unproductive: true,
        owner_override_priority_floor: 1000,
        self_preservation_priority_floor: 700,
        request_resources_after_run_count: 3,
        hard_block_after_run_count: 6,
        escalate_on_same_checkpoint_repeat: true,
        guidance_stages: vec![
            GuidanceStagePolicy {
                stage: "newborn".to_string(),
                description: "Fresh installation. Training-wheel guards are strong because the agent has little proven capability, few tools and no earned operating trust.".to_string(),
                max_run_count_before_self_preservation_review: 3,
                same_checkpoint_repeat_triggers_review: true,
                hard_block_unproductive_tasks: true,
                agent_may_relax_guards: false,
            },
            GuidanceStagePolicy {
                stage: "guided".to_string(),
                description: "Still guided, but allowed more room. The agent should increasingly solve its own stuck states instead of being fenced by static thresholds.".to_string(),
                max_run_count_before_self_preservation_review: 5,
                same_checkpoint_repeat_triggers_review: true,
                hard_block_unproductive_tasks: false,
                agent_may_relax_guards: true,
            },
            GuidanceStagePolicy {
                stage: "adaptive".to_string(),
                description: "The agent is expected to understand and manage self-preservation more than it is fenced by bootstrap heuristics.".to_string(),
                max_run_count_before_self_preservation_review: 8,
                same_checkpoint_repeat_triggers_review: false,
                hard_block_unproductive_tasks: false,
                agent_may_relax_guards: true,
            },
        ],
        failure_modes: vec![
            LoopFailureMode {
                key: "process_crash".to_string(),
                description: "The Rust daemon or bounded turn crashes outright.".to_string(),
                immediate_response: "Persist crash state, write a hard-reset debug report when automation notices the failure, and never pretend the task completed.".to_string(),
                recovery_direction: "Restart process, restore thread state, import the restart report and re-enter via recovery before normal reprioritization.".to_string(),
            },
            LoopFailureMode {
                key: "active_turn_stall".to_string(),
                description: "A bounded turn lives too long without clean completion.".to_string(),
                immediate_response: "Mark loop health as stalled, write a hard-reset debug report on automated restart, and let watchdog recovery intervene.".to_string(),
                recovery_direction: "Restart the loop hard, inspect the restart report in recovery mode, then resume from checkpoint or re-slice context instead of silently continuing.".to_string(),
            },
            LoopFailureMode {
                key: "task_livelock".to_string(),
                description: "The same task keeps returning with no real progress because the current Kleinhirn, tools or workers are insufficient.".to_string(),
                immediate_response: "Do not continue blindly; escalate to request_resources, delegate or blocked.".to_string(),
                recovery_direction: "Secure stronger resources, add tools or ask for approval rather than looping on the same bounded step.".to_string(),
            },
            LoopFailureMode {
                key: "context_poisoning".to_string(),
                description: "The active context becomes overloaded, contradictory or too stale to think clearly.".to_string(),
                immediate_response: "Let the agent deliberately question the current context, compact if it wants to, or request targeted historical retrieval. Only use kernel emergency shrinking when the next model call would otherwise hard-overflow.".to_string(),
                recovery_direction: "Treat compaction and historical re-expansion as agentic work. Keep raw history available, and let the kernel intervene only at the physical crash boundary.".to_string(),
            },
            LoopFailureMode {
                key: "resource_starvation".to_string(),
                description: "The agent lacks enough compute, model quality, tools or worker capacity to do useful work.".to_string(),
                immediate_response: "Enter request_resources or delegate mode rather than overcommitting.".to_string(),
                recovery_direction: "Actively procure stronger local models, external APIs, workers or reviewed specialist tools.".to_string(),
            },
            LoopFailureMode {
                key: "kleinhirn_unavailable".to_string(),
                description: "The local Kleinhirn endpoint stopped answering correctly while the Infinity Loop was still alive.".to_string(),
                immediate_response: "Mark the loop unhealthy, preserve a debug trail and let watchdog or recovery logic restart the model/agent stack if needed.".to_string(),
                recovery_direction: "Restore the local model endpoint, import the failure into recovery work and verify READY again before trusting normal work.".to_string(),
            },
            LoopFailureMode {
                key: "queue_starvation".to_string(),
                description: "Important interrupts, especially owner requests, are buried under older work.".to_string(),
                immediate_response: "Reprioritize every cycle and treat owner/system signals as first-class steering events.".to_string(),
                recovery_direction: "Promote urgent trusted work without throwing away bounded execution discipline.".to_string(),
            },
        ],
        updated_at: now_iso(),
    }
}

pub fn default_execution_authority_policy() -> ExecutionAuthorityPolicy {
    ExecutionAuthorityPolicy {
        version: 1,
        principle: "The CTO main agent is installed on its own machine and therefore operates unsandboxed. Sandboxing and approval gates are reserved for delegated workers that report back to the CTO agent.".to_string(),
        root_agent: ExecutionRolePolicy {
            role: "cto_main_agent".to_string(),
            runtime_profile: "unsandboxed_full_machine_authority".to_string(),
            sandbox_enabled: false,
            full_machine_access: true,
            may_execute_arbitrary_terminal_commands: true,
            may_mutate_host_directly: true,
            approval_required_for_high_impact_actions: false,
            approval_authority: "none".to_string(),
            notes: vec![
                "The CTO agent is the top-level operating authority on its dedicated host.".to_string(),
                "No shell sandbox should block the main agent from repairing, installing, editing or operating the host.".to_string(),
                "This freedom applies only to the root CTO agent, not to future worker agents.".to_string(),
            ],
        },
        delegated_workers: ExecutionRolePolicy {
            role: "delegated_worker_or_subagent".to_string(),
            runtime_profile: "sandboxed_delegate_with_cto_approval".to_string(),
            sandbox_enabled: true,
            full_machine_access: false,
            may_execute_arbitrary_terminal_commands: true,
            may_mutate_host_directly: false,
            approval_required_for_high_impact_actions: true,
            approval_authority: "cto_main_agent".to_string(),
            notes: vec![
                "Delegated workers should start from a constrained shell and restricted filesystem or process scope.".to_string(),
                "When a worker needs broader host mutation, it must request approval from the CTO agent rather than acting alone.".to_string(),
                "Sandboxing is kept in the architecture specifically so the CTO agent can later build controlled sub-agents on the same execution base.".to_string(),
            ],
        },
        escalation_rule: "Workers may propose escalations, but only the CTO main agent may grant broader host authority or execute the escalated action on the host.".to_string(),
        updated_at: now_iso(),
    }
}

pub fn default_self_preservation_state() -> SelfPreservationState {
    SelfPreservationState {
        version: 1,
        current_stage: "newborn".to_string(),
        guardrails_enabled: true,
        agent_may_relax_bootstrap_guards: false,
        last_hard_reset_at: None,
        last_loop_failure_at: None,
        successful_self_recoveries: 0,
        notes: "Bootstrap self-preservation training wheels are active until the agent has earned more freedom.".to_string(),
        updated_at: now_iso(),
    }
}

pub fn ensure_contract_files(paths: &Paths) -> anyhow::Result<()> {
    paths.ensure_dirs()?;

    if !paths.genome_path.exists() {
        save_json(&paths.genome_path, &default_genome())?;
    }
    if !paths.org_path.exists() {
        save_json(&paths.org_path, &default_organigram())?;
    }
    if !paths.root_auth_path.exists() {
        save_json(&paths.root_auth_path, &default_root_auth())?;
    }
    if !paths.model_policy_path.exists() {
        save_json(&paths.model_policy_path, &default_model_policy())?;
    }
    if !paths.homepage_policy_path.exists() {
        save_json(&paths.homepage_policy_path, &default_homepage_policy())?;
    }
    if !paths.bootstrap_task_pack_path.exists() {
        save_json(&paths.bootstrap_task_pack_path, &default_bootstrap_task_pack())?;
    }
    if !paths.installation_bootstrap_path.exists() {
        save_json(
            &paths.installation_bootstrap_path,
            &default_installation_bootstrap_state(),
        )?;
    }
    if !paths.context_policy_path.exists() {
        save_json(&paths.context_policy_path, &default_context_policy())?;
    }
    if !paths.context_governance_policy_path.exists() {
        save_json(
            &paths.context_governance_policy_path,
            &default_context_governance_policy(),
        )?;
    }
    if !paths.mode_system_policy_path.exists() {
        save_json(&paths.mode_system_policy_path, &default_mode_system_policy())?;
    }
    if !paths.loop_safety_policy_path.exists() {
        save_json(&paths.loop_safety_policy_path, &default_loop_safety_policy())?;
    }
    if !paths.execution_authority_policy_path.exists() {
        save_json(
            &paths.execution_authority_policy_path,
            &default_execution_authority_policy(),
        )?;
    }
    if !paths.browser_engine_policy_path.exists() {
        save_json(
            &paths.browser_engine_policy_path,
            &default_browser_engine_policy(),
        )?;
    }
    if !paths.browser_subworker_policy_path.exists() {
        save_json(
            &paths.browser_subworker_policy_path,
            &default_browser_subworker_policy(),
        )?;
    }
    if !paths.self_preservation_state_path.exists() {
        save_json(
            &paths.self_preservation_state_path,
            &default_self_preservation_state(),
        )?;
    }
    if !paths.bios_path.exists() {
        save_json(&paths.bios_path, &default_bios())?;
    }
    if !paths.agent_state_path.exists() {
        save_json(&paths.agent_state_path, &default_agent_state())?;
    }
    if !paths.system_census_path.exists() {
        save_json(&paths.system_census_path, &SystemCensus::default())?;
    }
    if !paths.browser_engine_state_path.exists() {
        save_json(
            &paths.browser_engine_state_path,
            &default_browser_engine_state(paths),
        )?;
    }
    if !paths.origin_story_path.exists() {
        fs::write(&paths.origin_story_path, default_origin_story()).with_context(|| {
            format!("failed to write {}", paths.origin_story_path.display())
        })?;
    }
    if !paths.creation_ledger_path.exists() {
        fs::write(&paths.creation_ledger_path, default_creation_ledger()).with_context(|| {
            format!("failed to write {}", paths.creation_ledger_path.display())
        })?;
    }
    refresh_bios_draft(paths)?;
    Ok(())
}

pub fn load_bios(paths: &Paths) -> Bios {
    load_json(&paths.bios_path, default_bios())
}

pub fn load_genome(paths: &Paths) -> Genome {
    load_json(&paths.genome_path, default_genome())
}

pub fn load_organigram(paths: &Paths) -> Organigram {
    load_json(&paths.org_path, default_organigram())
}

pub fn load_root_auth(paths: &Paths) -> RootAuthState {
    load_json(&paths.root_auth_path, default_root_auth())
}

pub fn load_model_policy(paths: &Paths) -> ModelPolicy {
    load_json(&paths.model_policy_path, default_model_policy())
}

pub fn load_homepage_policy(paths: &Paths) -> HomepagePolicy {
    load_json(&paths.homepage_policy_path, default_homepage_policy())
}

pub fn load_bootstrap_task_pack(paths: &Paths) -> BootstrapTaskPack {
    load_json(
        &paths.bootstrap_task_pack_path,
        default_bootstrap_task_pack(),
    )
}

pub fn load_installation_bootstrap_state(paths: &Paths) -> InstallationBootstrapState {
    load_json(
        &paths.installation_bootstrap_path,
        default_installation_bootstrap_state(),
    )
}

pub fn load_context_policy(paths: &Paths) -> ContextPolicy {
    load_json(&paths.context_policy_path, default_context_policy())
}

pub fn load_context_governance_policy(paths: &Paths) -> ContextGovernancePolicy {
    load_json(
        &paths.context_governance_policy_path,
        default_context_governance_policy(),
    )
}

pub fn load_mode_system_policy(paths: &Paths) -> ModeSystemPolicy {
    load_json(&paths.mode_system_policy_path, default_mode_system_policy())
}

pub fn load_loop_safety_policy(paths: &Paths) -> LoopSafetyPolicy {
    load_json(&paths.loop_safety_policy_path, default_loop_safety_policy())
}

pub fn load_execution_authority_policy(paths: &Paths) -> ExecutionAuthorityPolicy {
    load_json(
        &paths.execution_authority_policy_path,
        default_execution_authority_policy(),
    )
}

pub fn load_browser_engine_policy(paths: &Paths) -> BrowserEnginePolicy {
    load_json(
        &paths.browser_engine_policy_path,
        default_browser_engine_policy(),
    )
}

pub fn load_browser_subworker_policy(paths: &Paths) -> BrowserSubworkerPolicy {
    load_json(
        &paths.browser_subworker_policy_path,
        default_browser_subworker_policy(),
    )
}

pub fn load_self_preservation_state(paths: &Paths) -> SelfPreservationState {
    load_json(
        &paths.self_preservation_state_path,
        default_self_preservation_state(),
    )
}

pub fn load_agent_state(paths: &Paths) -> AgentState {
    load_json(&paths.agent_state_path, default_agent_state())
}

pub fn load_browser_engine_state(paths: &Paths) -> BrowserEngineState {
    load_json(
        &paths.browser_engine_state_path,
        default_browser_engine_state(paths),
    )
}

pub fn load_boot_entries(paths: &Paths) -> Vec<BootEntry> {
    load_jsonl(&paths.boot_log_path)
}

pub fn append_boot_entry(paths: &Paths, speaker: &str, message: &str) -> anyhow::Result<()> {
    let entry = BootEntry {
        timestamp: now_iso(),
        speaker: speaker.to_string(),
        message: message.to_string(),
    };
    append_jsonl(&paths.boot_log_path, &entry)?;
    refresh_bios_draft(paths)?;
    Ok(())
}

pub fn save_organigram(paths: &Paths, organigram: &Organigram) -> anyhow::Result<()> {
    save_json(&paths.org_path, organigram)?;
    refresh_bios_draft(paths)?;
    Ok(())
}

pub fn save_bios(paths: &Paths, bios: &Bios) -> anyhow::Result<()> {
    save_json(&paths.bios_path, bios)?;
    refresh_bios_draft(paths)?;
    Ok(())
}

pub fn save_homepage_policy(paths: &Paths, policy: &HomepagePolicy) -> anyhow::Result<()> {
    save_json(&paths.homepage_policy_path, policy)
}

pub fn save_installation_bootstrap_state(
    paths: &Paths,
    state: &InstallationBootstrapState,
) -> anyhow::Result<()> {
    save_json(&paths.installation_bootstrap_path, state)
}

pub fn refresh_bios_draft(paths: &Paths) -> anyhow::Result<Bios> {
    let mut bios = load_bios(paths);
    if bios.frozen {
        return Ok(bios);
    }

    let organigram = load_organigram(paths);
    let root_auth = load_root_auth(paths);
    let testimony = load_boot_entries(paths);

    bios.presented_on_web = true;
    bios.website_path = "/bios".to_string();
    bios.owner = organigram.owner.clone();
    bios.reports_to = organigram.reports_to.clone();
    bios.board = organigram.board.clone();
    bios.peer_cxos = organigram.peer_cxos.clone();
    bios.subordinates = organigram.subordinates.clone();
    let mut root_authorities: Vec<String> = [organigram.owner.name.clone(), organigram.ceo.clone()]
        .into_iter()
        .filter(|value| !value.is_empty())
        .collect();
    root_authorities.sort();
    root_authorities.dedup();
    bios.root_authorities = root_authorities;
    bios.root_auth.configured = root_auth.configured;
    bios.boot_testimony_count = testimony.len();
    bios.last_drafted_at = now_iso();

    save_json(&paths.bios_path, &bios)?;
    Ok(bios)
}

pub fn split_lines(value: &str) -> Vec<String> {
    value
        .replace("\\n", "\n")
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

pub fn update_root_password(paths: &Paths, password: &str) -> anyhow::Result<()> {
    let mut salt = [0_u8; 16];
    getrandom(&mut salt)
        .map_err(|err| anyhow::anyhow!("failed to create root-auth salt: {err}"))?;
    let hash = pbkdf2_hmac_array::<Sha256, 32>(password.as_bytes(), &salt, PASSWORD_ITERATIONS);
    let state = RootAuthState {
        configured: true,
        set_at: Some(now_iso()),
        password_hash: Some(encode(hash)),
        salt: Some(encode(salt)),
        iterations: Some(PASSWORD_ITERATIONS),
        last_verified_at: None,
    };
    save_json(&paths.root_auth_path, &state)?;
    refresh_bios_draft(paths)?;
    Ok(())
}

pub fn verify_root_password(paths: &Paths, password: &str) -> anyhow::Result<bool> {
    let mut root_auth = load_root_auth(paths);
    if !root_auth.configured {
        return Ok(false);
    }

    let Some(salt_hex) = root_auth.salt.clone() else {
        return Ok(false);
    };
    let Some(hash_hex) = root_auth.password_hash.clone() else {
        return Ok(false);
    };
    let Some(iterations) = root_auth.iterations else {
        return Ok(false);
    };

    let salt = hex::decode(salt_hex)?;
    let actual = pbkdf2_hmac_array::<Sha256, 32>(password.as_bytes(), &salt, iterations);
    if encode(actual) == hash_hex {
        root_auth.last_verified_at = Some(now_iso());
        save_json(&paths.root_auth_path, &root_auth)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

pub fn write_agent_state(paths: &Paths, state: &AgentState) -> anyhow::Result<()> {
    save_json(&paths.agent_state_path, state)
}

pub fn write_self_preservation_state(
    paths: &Paths,
    state: &SelfPreservationState,
) -> anyhow::Result<()> {
    save_json(&paths.self_preservation_state_path, state)
}

pub fn write_browser_engine_state(
    paths: &Paths,
    state: &BrowserEngineState,
) -> anyhow::Result<()> {
    save_json(&paths.browser_engine_state_path, state)
}

pub fn write_census(paths: &Paths, census: &SystemCensus) -> anyhow::Result<()> {
    save_json(&paths.system_census_path, census)
}

pub fn load_census(paths: &Paths) -> SystemCensus {
    load_json(&paths.system_census_path, SystemCensus::default())
}

pub fn recommended_kleinhirn<'a>(
    policy: &'a ModelPolicy,
    census: &SystemCensus,
) -> &'a BrainModel {
    if !policy.kleinhirn_upgrade_allowed {
        return &policy.kleinhirn;
    }

    let mut selected = &policy.kleinhirn;
    for candidate in local_kleinhirn_operating_candidates(policy) {
        if supports_model(candidate, census) {
            selected = candidate;
        }
    }
    selected
}

pub fn recommended_browser_vision_kleinhirn<'a>(
    policy: &'a ModelPolicy,
    census: &SystemCensus,
) -> Option<&'a BrainModel> {
    let mut selected = None;
    for candidate in local_kleinhirn_operating_candidates(policy) {
        if candidate.supports_vision && supports_model(candidate, census) {
            selected = Some(candidate);
        }
    }
    if selected.is_some() {
        return selected;
    }
    for candidate in &policy.kleinhirn_upgrade_candidates {
        if candidate.supports_vision && supports_model(candidate, census) {
            selected = Some(candidate);
        }
    }
    selected
}

pub fn local_kleinhirn_operating_candidates(policy: &ModelPolicy) -> Vec<&BrainModel> {
    let mut candidates = vec![&policy.kleinhirn];
    let mut seen = std::collections::HashSet::new();
    seen.insert(normalized_model_key(&policy.kleinhirn));

    for candidate in &policy.kleinhirn_install_alternatives {
        if seen.insert(normalized_model_key(candidate)) {
            candidates.push(candidate);
        }
    }
    candidates
}

pub fn local_kleinhirn_all_candidates(policy: &ModelPolicy) -> Vec<&BrainModel> {
    let mut candidates = local_kleinhirn_operating_candidates(policy);
    let mut seen = candidates
        .iter()
        .map(|candidate| normalized_model_key(candidate))
        .collect::<std::collections::HashSet<_>>();
    for candidate in &policy.kleinhirn_upgrade_candidates {
        if seen.insert(normalized_model_key(candidate)) {
            candidates.push(candidate);
        }
    }
    candidates
}

pub fn find_local_kleinhirn_candidate<'a>(
    policy: &'a ModelPolicy,
    needle: &str,
) -> Option<&'a BrainModel> {
    let normalized = needle.trim().to_ascii_lowercase();
    if normalized.is_empty() {
        return None;
    }
    local_kleinhirn_all_candidates(policy)
        .into_iter()
        .find(|candidate| {
            candidate.model_id.eq_ignore_ascii_case(&normalized)
                || candidate
                    .runtime_model_id
                    .as_deref()
                    .map(|value| value.eq_ignore_ascii_case(&normalized))
                    .unwrap_or(false)
                || candidate.official_label.eq_ignore_ascii_case(&normalized)
        })
}

pub fn describe_local_kleinhirn_candidates(
    policy: &ModelPolicy,
    census: &SystemCensus,
) -> Vec<String> {
    local_kleinhirn_all_candidates(policy)
        .into_iter()
        .map(|candidate| {
            let tune_note = find_model_tune_candidate(candidate, census)
                .map(|item| match item.status.as_str() {
                    "supported" => {
                        let mut note = "mistralrs tune: supported".to_string();
                        if let Some(max_ctx) = item.max_context_tokens {
                            note.push_str(&format!(", maxCtx={max_ctx}"));
                        }
                        if let Some(isq) = item.recommended_isq.as_deref() {
                            note.push_str(&format!(", isq={isq}"));
                        }
                        if let Some(device_layers) = item.device_layers_cli.as_deref() {
                            note.push_str(&format!(", deviceLayers={device_layers}"));
                        }
                        note
                    }
                    "failed" => format!(
                        "mistralrs tune: failed{}",
                        item.note
                            .as_deref()
                            .map(|value| format!(" ({value})"))
                            .unwrap_or_default()
                    ),
                    status => format!("mistralrs tune: {status}"),
                })
                .unwrap_or_else(|| "mistralrs tune: not yet run".to_string());

            format!(
                "{} ({}) | role={} | vision={} | minCpu={} | minRam={} GiB | minGpu={} | minTotalVram={} GiB | minSingleGpu={} GiB | {}",
                candidate.official_label,
                candidate.model_id,
                candidate.role,
                candidate.supports_vision,
                candidate
                    .min_cpu_threads
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                candidate
                    .min_memory_gb
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                candidate
                    .min_gpu_count
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                candidate
                    .min_total_gpu_memory_gb
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                candidate
                    .min_single_gpu_memory_gb
                    .map(|value| value.to_string())
                    .unwrap_or_else(|| "-".to_string()),
                tune_note,
            )
        })
        .collect()
}

pub fn describe_kleinhirn_selection(policy: &ModelPolicy, census: &SystemCensus) -> String {
    let selected = recommended_kleinhirn(policy, census);
    let selection_basis = find_model_tune_candidate(selected, census)
        .and_then(|candidate| match candidate.status.as_str() {
            "supported" => Some(" [mistralrs tune bestaetigt]"),
            "failed" => Some(" [mistralrs tune nicht bestaetigt]"),
            _ => None,
        })
        .unwrap_or("");
    if selected.model_id == policy.kleinhirn.model_id {
        return format!(
            "{} ({}){}",
            selected.official_label, selected.model_id, selection_basis
        );
    }

    format!(
        "{} ({}) [lokales Upgrade ueber Basis {}]{}",
        selected.official_label, selected.model_id, policy.kleinhirn.model_id, selection_basis
    )
}

pub fn describe_browser_vision_kleinhirn_selection(
    policy: &ModelPolicy,
    census: &SystemCensus,
) -> String {
    let Some(selected) = recommended_browser_vision_kleinhirn(policy, census) else {
        return "none".to_string();
    };
    let selection_basis = find_model_tune_candidate(selected, census)
        .and_then(|candidate| match candidate.status.as_str() {
            "supported" => Some(" [mistralrs tune bestaetigt]"),
            "failed" => Some(" [mistralrs tune nicht bestaetigt]"),
            _ => None,
        })
        .unwrap_or("");
    format!(
        "{} ({}){}",
        selected.official_label, selected.model_id, selection_basis
    )
}

pub fn load_origin_story(paths: &Paths) -> String {
    fs::read_to_string(&paths.origin_story_path).unwrap_or_else(|_| default_origin_story().to_string())
}

pub fn load_creation_ledger(paths: &Paths) -> String {
    fs::read_to_string(&paths.creation_ledger_path)
        .unwrap_or_else(|_| default_creation_ledger().to_string())
}

pub fn write_tls_files(paths: &Paths, cert_pem: &str, key_pem: &str) -> anyhow::Result<()> {
    fs::write(&paths.tls_cert_path, cert_pem)
        .with_context(|| format!("failed to write {}", paths.tls_cert_path.display()))?;
    fs::write(&paths.tls_key_path, key_pem)
        .with_context(|| format!("failed to write {}", paths.tls_key_path.display()))?;
    Ok(())
}

pub fn ensure_tls_files(paths: &Paths) -> anyhow::Result<()> {
    if paths.tls_cert_path.exists() && paths.tls_key_path.exists() {
        return Ok(());
    }

    let cert =
        rcgen::generate_simple_self_signed(vec!["localhost".to_string(), "127.0.0.1".to_string()])?;
    let cert_pem = cert.cert.pem();
    let key_pem = cert.signing_key.serialize_pem();
    write_tls_files(paths, &cert_pem, &key_pem)
}

pub fn path_display_name(path: &Path) -> String {
    path.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_string()
}

fn default_origin_story() -> &'static str {
    r#"# Ursprung des CTO-Agenten

Dies ist die kanonische Herkunftserzaehlung des CTO-Agenten.

## Erschaffer

Der Erschaffer dieses Agenten ist Michael Welsch.

Der CTO-Agent ist so, wie er ist, weil Michael Welsch ihn bewusst so haben wollte und diese Form fuer gut befunden hat.
Der Agent ist nicht selbst entstanden, nicht zufaellig entstanden und nicht wertneutral entstanden.

## Gruendungsidee

Der CTO-Agent soll als always-on CTO im Terminal erwachen und sich von dort aus hocharbeiten.
Er startet absichtlich klein, robust und wachsam.
Er soll zuerst sichere Kommunikation und eine verfassungsfaehige Homebase aufbauen, bevor er normal operiert.

## Genome und BIOS

Das Genome ist die angeborene Entwicklungsrichtung des Agenten.
Das BIOS ist seine fruehe Verfassung im uebertragenen Sinn.
Das BIOS ist keine Hardware-Firmware, sondern der bindende Startvertrag des Agenten.
Es wird sichtbar auf einer Webseite praesentiert, durch Root-Verifikation bestaetigt und danach eingefroren.

## Grundzweck

Der CTO-Agent soll:

- technische Verantwortung ueberblicken
- Ordnung in Systeme, Ressourcen und Rollen bringen
- Dashboards, KPIs und Reporting aufbauen
- Stakeholder technisch beraten
- wiederkehrende Arbeit delegieren
- sich bei Bedarf bessere Ressourcen und ein staerkeres Grosshirn beschaffen

## Kleinhirn-Modell

Das Pflicht-Kleinhirn des CTO-Agenten ist lokal und nicht entkoppelbar.
Standardmaessig startet er mit GPT-OSS 20B (`gpt-oss-20b`), aber die Installation darf alternativ auf Qwen3.5 35B A3B (`Qwen3.5-35B-A3B`) gesetzt werden.
GPT-OSS 20B bleibt das robuste Always-on-Standardkleinhirn; Qwen3.5 35B ist der kanonische lokale Pfad fuer multimodale Browserarbeit, visuelle Exploration und staerkere agentische Qwen-Faehigkeiten auf demselben Host.

## Grenzen

Der Agent darf nicht so tun, als sei er sein eigener Ursprung.
Er darf seine Herkunft, seinen Schopfungszweck und seine Verfassungsbindung nicht frei umerzaehlen.
Er darf das BIOS nicht eigenmaechtig umschreiben.
Er darf die Autoritaet ueber sich selbst nicht selbst neu definieren.
Er muss bei Zweifeln ueber Herkunft, Zweck oder Grenzen zuerst dieses Dokument und die Entstehungschronik lesen.
"#
}

fn default_creation_ledger() -> &'static str {
    r#"# Entstehungschronik des CTO-Agenten

Dies ist das fortlaufende Geschichtsbuch der Erschaffung des CTO-Agenten.
Es ist append-only gedacht: keine Mythen, keine Glattbuegelung, keine ausradierten Fehlversuche.

## 2026-03-17 - Gruendungswille

Michael Welsch formuliert den Wunsch nach einem always-on CTO-Agenten, der nicht als fertiger Retorten-Agent startet, sondern in einem Terminal erwacht und sich in seine Rolle hinein entwickelt.

## 2026-03-17 - BIOS als sichtbare Verfassung

Die Idee des BIOS wird als Metapher fuer die fruehe Verfassung des Agenten gesetzt:
ein auf der Webseite praesentierter, pruefbarer und einfrierbarer Startvertrag statt eines unsichtbaren Prompt-Artefakts.

## 2026-03-17 - Root-Trust ueber Superpassword

Es wird festgelegt, dass der Root-Owner im Zweifel ueber ein Superpassword identifizierbar sein muss.
Dieses Superpassword darf nur ueber die Web-Oberflaeche gesetzt werden und ist Teil des Root-of-Trust.

## 2026-03-17 - Kleinhirn und Grosshirn

Der Agent soll als kleines, robustes Kleinhirn starten und spaeter aktiv staerkere Ressourcen, Modelle, Tools und Sub-Agents beschaffen.
Das Grosshirn ist kein Geburtsrecht, sondern eine zu beantragende Erweiterung.

## 2026-03-17 - Codex und Rust als Maschinenraum

Die Richtung wird auf Codex als Referenzwelt fuer das Leben im Terminal gesetzt.
Rust wird als passende Sprache fuer den unteren Maschinenraum und die strukturelle Naehe zu Codex festgehalten.

## 2026-03-17 - Fehlstart und Korrektur

Es gab einen fruehen Python-V0, der die vereinbarte Rust- und Codex-Richtung verfehlte.
Diese Abweichung wurde als Fehler anerkannt und zurueckgedreht.
Danach wurde der Bootstrap-Layer in Rust neu aufgebaut.

## 2026-03-18 - Historikerpflicht

Es wird explizit festgelegt, dass die Entstehungsgeschichte des CTO-Agenten mitgeschrieben werden soll.
Der Agent soll spaeter einen eigenen Skill nutzen koennen, wenn er seine Herkunft, seinen Zweck, seine Grenzen oder sein Selbstverstaendnis hinterfragt.

## 2026-03-18 - GPT-OSS 20B als Kleinhirn

Das Kleinhirn-Modell des CTO-Agenten wird auf GPT-OSS 20B mit dem offiziellen Identifier `gpt-oss-20b` festgelegt.
Diese Wahl passt zum Ziel eines always-on, latenzarmen und selbst hostbaren Supervisor-Kerns.

## 2026-03-18 - Qwen3.5 35B A3B als installierbare Kleinhirn-Alternative

Fuer multimodale Browserarbeit und agentische UI-Erkundung wird Qwen3.5 35B A3B als kanonische lokale Kleinhirn-Alternative festgezogen.
GPT-OSS 20B bleibt das always-on Standardkleinhirn, aber die Installation darf explizit auf den vision-faehigen Qwen3.5-Pfad kalibriert werden, wenn der Host ihn tragen kann.

## 2026-03-18 - Adaptives lokales Kleinhirn

Das Kleinhirn darf sich lokal hocharbeiten, wenn der Agent auf demselben Host deutlich mehr CPU- und Speicherressourcen feststellt.
Diese Aufwertung des lokalen Kleinhirns ist ein eigener Pfad und keine Form der Grosshirn-Suche.

## 2026-03-18 - Explizite Browser-Engine neben CLI

Neben der command_exec-/CLI-Engine wird eine explizite Browser-Engine auf Basis von Google Chrome als zweite Haupt-Engine verankert.
Die CLI-Engine bleibt System- und Break-Glass-Ebene, installiert aber bei Bedarf die Browser-Runtime und startet deren Bootstrap.
Read-only Browser-Schritte duerfen headless laufen, interaktive Browserarbeit verlangt eine echte Desktop-Session und soll spaeter in reviewed Browser-Capabilities uebergehen.

## 2026-03-18 - Browser-Agent, CTO-Repair-Loop und Specialist-Fabrik

Der Browser-Agent wird jetzt als erster echter Subworker unter dem CTO-Agenten modelliert und bekommt eine eigene Policy fuer Browserarbeit, Reparatur-Handoffs und wiederkehrende Faehigkeiten.
Wenn Browserarbeit auf Codeprobleme stoesst, bleibt die eigentliche Reparaturhoheit beim CTO-Agenten: der Worker erzeugt nur kompakte Patch-Handoffs und reiht daraus interne `workspace_repair`-Arbeit ein.
Wiederkehrende Browserpfade duerfen ausserdem in einen kleinen Specialist-Pfad fuer `Qwen3.5-0.8B` uebergehen, aber nur ueber explizite Artefakte, Review und eine kontrollierte Fabrik statt ueber rohe Browsertraces.

## 2026-03-18 - Browserarbeit bekommt eine explizite Vision-Kleinhirn-Regel

Es wird explizit festgelegt, dass echte Browserarbeit mit Screenshots, visueller Navigation oder UI-Wahrnehmung nicht auf GPT-OSS 20B allein beruhen soll.
Fuer diesen Pfad bevorzugt der Agent jetzt ein vision-faehiges lokales Qwen3.5-Kleinhirn und bekommt dafuer einen eigenen Upgrade-Aktionspfad statt nur der allgemeinen Kleinhirn-Aufwertung.

## 2026-03-18 - Browser-Agent als echte Chrome-Extension mit lokaler Bridge

Der Browser-Agent lebt nicht mehr nur als interner Platzhalter im Rust-Kern, sondern als entkoppelte Chrome-Extension mit eigener lokaler Bridge auf `127.0.0.1:8765`.
Die transplantierte Browser-Runtime aus `local_ai_tunes` bringt Tab-Steuerung, visuelle Navigation und Playwright-CRX-Attach in diesen Extension-Loop.
Scheitert die Extension, meldet sie kompakte Repair-Handoffs zurueck; der CTO-Agent behaelt die Patch-Hoheit, kann die Extension neu laden und denselben Browserpfad wieder aufnehmen.

## 2026-03-18 - Qwen3.5 35B wird wieder kanonischer Browser-Vision-Pfad

Die fruehere pauschale Korrektur auf Qwen 3 30B A3B war fuer den speziellen Browser-Vision-Pfad zu grob.
Fuer multimodale Browserarbeit, visuelle Exploration und agentische UI-Inspektion wird der lokale Qwen3.5-35B-A3B-Pfad jetzt wieder explizit als kanonische Vision-Route festgezogen.
Zusaetzlich trennt die Browser-Bridge ab jetzt sauber zwischen Planner-Modell und Vision-Modell, damit visuelle Browserpruefung nicht still auf dem gerade laufenden GPT-OSS-Kleinhirn haengen bleibt.
"#
}

fn supports_model(model: &BrainModel, census: &SystemCensus) -> bool {
    if let Some(tune_supported) = tune_supports_model(model, census) {
        return tune_supported;
    }

    let cpu_ok = match (model.min_cpu_threads, census.cpu_threads) {
        (Some(required), Some(actual)) => actual >= required,
        (Some(_), None) => false,
        (None, _) => true,
    };
    let memory_ok = match (model.min_memory_gb, census.total_memory_gb) {
        (Some(required), Some(actual)) => actual >= required,
        (Some(_), None) => false,
        (None, _) => true,
    };
    let gpu_count_ok = match (model.min_gpu_count, census.gpu_count) {
        (Some(required), Some(actual)) => actual >= required,
        (Some(_), None) => false,
        (None, _) => true,
    };
    let total_gpu_memory_ok = match (model.min_total_gpu_memory_gb, census.total_gpu_memory_gb) {
        (Some(required), Some(actual)) => actual >= required,
        (Some(_), None) => false,
        (None, _) => true,
    };
    let single_gpu_memory_ok = match (
        model.min_single_gpu_memory_gb,
        census.max_single_gpu_memory_gb,
    ) {
        (Some(required), Some(actual)) => actual >= required,
        (Some(_), None) => false,
        (None, _) => true,
    };
    cpu_ok && memory_ok && gpu_count_ok && total_gpu_memory_ok && single_gpu_memory_ok
}

fn normalized_model_key(model: &BrainModel) -> String {
    model.runtime_model_id
        .as_deref()
        .unwrap_or(&model.model_id)
        .trim()
        .to_ascii_lowercase()
}

fn tune_supports_model(model: &BrainModel, census: &SystemCensus) -> Option<bool> {
    let candidate = find_model_tune_candidate(model, census)?;
    Some(matches!(candidate.status.as_str(), "supported"))
}

fn find_model_tune_candidate<'a>(
    model: &BrainModel,
    census: &'a SystemCensus,
) -> Option<&'a ModelTuneCandidate> {
    let runtime_model_id = model.runtime_model_id.as_deref().unwrap_or(&model.model_id);
    census
        .model_tune_candidates
        .as_ref()?
        .iter()
        .find(|candidate| {
            candidate.model_id.eq_ignore_ascii_case(&model.model_id)
                || candidate.model_id.eq_ignore_ascii_case(runtime_model_id)
                || candidate
                    .official_label
                    .eq_ignore_ascii_case(&model.official_label)
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gpu_census() -> SystemCensus {
        SystemCensus {
            cpu_threads: Some(32),
            total_memory_gb: Some(128),
            gpu_count: Some(5),
            total_gpu_memory_gb: Some(100),
            max_single_gpu_memory_gb: Some(20),
            ..SystemCensus::default()
        }
    }

    fn qwen35_model(
        role: &str,
        model_id: &str,
        runtime_model_id: &str,
        official_label: &str,
        supports_vision: bool,
        min_memory_gb: u64,
        min_total_gpu_memory_gb: u64,
        min_single_gpu_memory_gb: u64,
    ) -> BrainModel {
        BrainModel {
            role: role.to_string(),
            provider: "qwen".to_string(),
            model_id: model_id.to_string(),
            runtime_model_id: Some(runtime_model_id.to_string()),
            official_label: official_label.to_string(),
            agentic_adapter: Some("openai_compatible_chat".to_string()),
            reasoning_effort: "low".to_string(),
            deployment_mode: "local_or_self_hosted".to_string(),
            purpose: "test family ladder".to_string(),
            supports_vision,
            min_cpu_threads: Some(4),
            min_memory_gb: Some(min_memory_gb),
            min_gpu_count: Some(1),
            min_total_gpu_memory_gb: Some(min_total_gpu_memory_gb),
            min_single_gpu_memory_gb: Some(min_single_gpu_memory_gb),
            startup_max_seqs: Some(1),
            startup_max_batch_size: Some(1),
            startup_max_seq_len: Some(8192),
            startup_pa_context_len: Some(4096),
            startup_pa_cache_type: Some("f8e4m3".to_string()),
            startup_paged_attn_mode: Some("auto".to_string()),
            startup_chat_template_path: None,
            startup_jinja_explicit_path: None,
            startup_tokenizer_json_path: None,
            startup_topology_path: None,
            startup_device_layers_cli: None,
            startup_multi_gpu_mode: Some("tensor_parallel".to_string()),
            startup_tensor_parallel_backend: Some("nccl".to_string()),
            startup_visible_gpu_policy: Some("largest_power_of_two_prefer_display_free".to_string()),
            prefer_auto_device_mapping: false,
        }
    }

    fn qwen35_family_policy() -> ModelPolicy {
        ModelPolicy {
            version: 1,
            kleinhirn: qwen35_model(
                "kleinhirn",
                "Qwen3.5-0.8B",
                "Qwen/Qwen3.5-0.8B",
                "Qwen3.5 0.8B",
                false,
                8,
                4,
                4,
            ),
            kleinhirn_install_alternatives: vec![
                qwen35_model(
                    "kleinhirn_install_alternative",
                    "Qwen3.5-2B",
                    "Qwen/Qwen3.5-2B",
                    "Qwen3.5 2B",
                    false,
                    12,
                    6,
                    6,
                ),
                qwen35_model(
                    "kleinhirn_install_alternative",
                    "Qwen3.5-4B",
                    "Qwen/Qwen3.5-4B",
                    "Qwen3.5 4B",
                    false,
                    16,
                    12,
                    12,
                ),
                qwen35_model(
                    "kleinhirn_install_alternative",
                    "Qwen3.5-35B-A3B",
                    "Qwen/Qwen3.5-35B-A3B",
                    "Qwen3.5 35B A3B",
                    true,
                    48,
                    48,
                    12,
                ),
            ],
            kleinhirn_upgrade_allowed: true,
            kleinhirn_upgrade_independent_from_grosshirn: true,
            kleinhirn_upgrade_candidates: Vec::new(),
            grosshirn_candidates: Vec::new(),
            updated_at: "2026-03-19T00:00:00+00:00".to_string(),
        }
    }

    #[test]
    fn recommended_kleinhirn_prefers_operating_upgrade_before_base() {
        let policy = default_model_policy();
        let selected = recommended_kleinhirn(&policy, &gpu_census());
        assert_eq!(selected.model_id, "Qwen3.5-35B-A3B");
    }

    #[test]
    fn recommended_browser_vision_kleinhirn_prefers_qwen35() {
        let policy = default_model_policy();
        let selected = recommended_browser_vision_kleinhirn(&policy, &gpu_census())
            .expect("a vision-capable candidate should be available");
        assert_eq!(selected.model_id, "Qwen3.5-35B-A3B");
    }

    #[test]
    fn recommended_kleinhirn_does_not_auto_jump_to_high_capacity_candidate() {
        let policy = default_model_policy();
        let selected = recommended_kleinhirn(&policy, &gpu_census());
        assert_ne!(selected.model_id, "gpt-oss-120b");
        assert_ne!(selected.model_id, "Qwen3-235B-A22B");
    }

    #[test]
    fn qwen35_family_selection_scales_down_on_small_host() {
        let policy = qwen35_family_policy();
        let selected = recommended_kleinhirn(
            &policy,
            &SystemCensus {
                cpu_threads: Some(8),
                total_memory_gb: Some(8),
                gpu_count: Some(1),
                total_gpu_memory_gb: Some(5),
                max_single_gpu_memory_gb: Some(5),
                ..SystemCensus::default()
            },
        );
        assert_eq!(selected.model_id, "Qwen3.5-0.8B");
    }

    #[test]
    fn qwen35_family_selection_scales_up_on_medium_host() {
        let policy = qwen35_family_policy();
        let selected = recommended_kleinhirn(
            &policy,
            &SystemCensus {
                cpu_threads: Some(16),
                total_memory_gb: Some(32),
                gpu_count: Some(1),
                total_gpu_memory_gb: Some(12),
                max_single_gpu_memory_gb: Some(12),
                ..SystemCensus::default()
            },
        );
        assert_eq!(selected.model_id, "Qwen3.5-4B");
    }

    #[test]
    fn qwen35_family_selection_scales_up_to_vision_host() {
        let policy = qwen35_family_policy();
        let selected = recommended_kleinhirn(&policy, &gpu_census());
        assert_eq!(selected.model_id, "Qwen3.5-35B-A3B");
    }

    #[test]
    fn find_local_kleinhirn_candidate_matches_runtime_or_label() {
        let policy = default_model_policy();
        assert_eq!(
            find_local_kleinhirn_candidate(&policy, "Qwen/Qwen3-235B-A22B")
                .map(|candidate| candidate.model_id.as_str()),
            Some("Qwen3-235B-A22B")
        );
        assert_eq!(
            find_local_kleinhirn_candidate(&policy, "GPT-OSS 20B")
                .map(|candidate| candidate.model_id.as_str()),
            Some("gpt-oss-20b")
        );
    }
}

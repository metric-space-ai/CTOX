use crate::browser_engine::BrowserActionDirective;
use crate::contracts::load_census;
use crate::contracts::Paths;
use crate::contracts::load_model_policy;
use crate::contracts::recommended_kleinhirn;
use crate::runtime_db::LearningEntryDraft;
use crate::runtime_db::ProactiveContactDraft;
use crate::runtime_db::ProactiveContactValidationDraft;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::task_has_active_grosshirn_boost;
use crate::runtime_db::list_queued_tasks;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::latest_context_package_for_task;
use crate::runtime_db::record_resource_status;
use crate::runtime_db::select_next_task;
use crate::tooling::ExecCommandDirective;
use crate::tooling::HomepageUpdateDirective;
use anyhow::Context;
use reqwest::blocking::Client;
use serde::Deserialize;
use serde_json::Map;
use serde_json::Value;
use std::collections::BTreeMap;
use std::fs;
use std::io::Read;
use std::io::Write;
use std::net::TcpStream;
use std::time::Duration;

#[derive(Debug, Clone, Deserialize)]
pub struct AgenticRunResult {
    pub status: String,
    pub reply: Option<String>,
    pub final_output: Option<String>,
    pub blocked_reason: Option<String>,
    pub model: Option<String>,
    pub task_status: Option<String>,
    pub next_mode: Option<String>,
    pub checkpoint_summary: Option<String>,
    pub checkpoint_detail: Option<String>,
    pub context_directive: Option<ContextDirective>,
    pub system_census_action: Option<String>,
    pub brain_directive: Option<BrainDirective>,
    pub exec_session_directive: Option<ExecSessionDirective>,
    pub exec_directive: Option<ExecCommandDirective>,
    pub browser_directive: Option<BrowserActionDirective>,
    pub homepage_update: Option<HomepageUpdateDirective>,
    pub delegate_contract: Option<DelegationContract>,
    pub followup_task: Option<FollowupTaskDirective>,
    pub learning_entries: Vec<LearningEntryDraft>,
    pub proactive_contact_draft: Option<ProactiveContactDraft>,
    pub proactive_contact_validation: Option<ProactiveContactValidationDraft>,
    pub used_grosshirn: bool,
    pub fell_back_to_kleinhirn: bool,
    pub model_usage: Option<ModelUsageMetrics>,
}

impl AgenticRunResult {
    pub fn blocked(reason: impl Into<String>) -> Self {
        Self {
            status: "blocked".to_string(),
            reply: None,
            final_output: None,
            blocked_reason: Some(reason.into()),
            model: None,
            task_status: None,
            next_mode: None,
            checkpoint_summary: None,
            checkpoint_detail: None,
            context_directive: None,
            system_census_action: None,
            brain_directive: None,
            exec_session_directive: None,
            exec_directive: None,
            browser_directive: None,
            homepage_update: None,
            delegate_contract: None,
            followup_task: None,
            learning_entries: Vec::new(),
            proactive_contact_draft: None,
            proactive_contact_validation: None,
            used_grosshirn: false,
            fell_back_to_kleinhirn: false,
            model_usage: None,
        }
    }

    pub fn best_reply(&self) -> Option<&str> {
        self.reply
            .as_deref()
            .or(self.final_output.as_deref())
    }

    pub fn status_note(&self) -> String {
        match (&self.status[..], self.blocked_reason.as_deref(), self.model.as_deref()) {
            ("ok", _, Some(model)) => format!("ok via {}", model),
            ("ok", _, None) => "ok".to_string(),
            (_, Some(reason), Some(model)) => format!("{} via {} ({})", self.status, model, reason),
            (_, Some(reason), None) => format!("{} ({})", self.status, reason),
            (_, None, Some(model)) => format!("{} via {}", self.status, model),
            _ => self.status.clone(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct DelegationContract {
    pub worker_kind: String,
    pub contract_title: String,
    pub contract_detail: String,
    pub request_note: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct FollowupTaskDirective {
    pub task_kind: String,
    pub title: String,
    pub detail: String,
    pub priority_score: i64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ContextDirective {
    pub action: String,
    pub concern: Option<String>,
    pub history_research_query: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct BrainDirective {
    pub action: String,
    pub target_model: Option<String>,
    pub note: Option<String>,
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct ModelUsageMetrics {
    pub input_tokens: Option<i64>,
    pub output_tokens: Option<i64>,
    pub total_tokens: Option<i64>,
    pub estimated_cost_usd: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ExecSessionDirective {
    pub action: String,
    pub session_id: Option<String>,
    pub command: Vec<String>,
    pub input: Option<String>,
    pub workdir: Option<String>,
    pub timeout_ms: Option<u64>,
    pub tty: bool,
    pub close_stdin: bool,
    pub rows: Option<u16>,
    pub cols: Option<u16>,
    pub justification: Option<String>,
}

#[derive(Debug, Clone)]
struct ModelTarget {
    base_url: String,
    model_id: String,
    api_key: String,
    adapter: String,
    brain_tier: String,
    source_label: String,
}

#[derive(Debug, Clone)]
struct ResolvedTargets {
    primary: ModelTarget,
    fallback: Option<ModelTarget>,
}

#[derive(Debug, Clone)]
struct ParsedHttpBaseUrl {
    host: String,
    port: u16,
    base_path: String,
}

pub fn run_agentic_task_once(
    paths: &Paths,
    reason: &str,
    task: &TaskRecord,
) -> anyhow::Result<AgenticRunResult> {
    let Some(resolved) = resolve_operating_targets(paths, task)? else {
        return Ok(AgenticRunResult::blocked(
            "missing_kleinhirn_endpoint: set CTO_AGENT_KLEINHIRN_BASE_URL or run a local OpenAI-compatible Kleinhirn endpoint",
        ));
    };
    let target = resolved.primary.clone();

    record_resource_status(paths, "agentic_loop", "target_model", "policy", &target.model_id)?;
    record_resource_status(paths, "agentic_loop", "model_endpoint", "ready", &target.base_url)?;
    record_resource_status(paths, "agentic_loop", "brain_tier", "policy", &target.brain_tier)?;
    record_resource_status(paths, "agentic_loop", "brain_source", "policy", &target.source_label)?;
    if let Some(fallback) = resolved.fallback.as_ref() {
        let _ = record_resource_status(
            paths,
            "agentic_loop",
            "brain_fallback",
            "policy",
            &format!("{} ({})", fallback.source_label, fallback.model_id),
        );
    }

    let context_block = latest_context_package_for_task(paths, task.id)?
        .map(|package| package.package_json)
        .unwrap_or_else(|| "{}".to_string());

    let system_prompt = build_system_prompt();
    let prompt_plan = build_prompt_plan(paths, reason, task, &system_prompt, &context_block)?;
    match post_model_request_with_fallback(paths, &resolved, &system_prompt, &prompt_plan.user_prompt) {
        Ok(response) => {
            let (used_target, response) = response;
            let content = match require_model_text_output(&used_target, &response) {
                Ok(content) => content,
                Err(err) if looks_like_empty_text_output_error(&err.to_string()) => {
                    let detail = err.to_string();
                    record_resource_status(
                        paths,
                        "agentic_loop",
                        "status",
                        "empty_text_output",
                        &detail,
                    )?;
                    return Ok(build_empty_text_retry_result(
                        &resolved,
                        &used_target,
                        &response,
                        &detail,
                        "Modell lieferte leeren Text; bounded Retry statt Hard-Block.",
                        "Kleinhirn lieferte keinen auswertbaren Text. Aufgabe wird erneut eingeordnet, statt hart zu blockieren.",
                    ));
                }
                Err(err) => return Err(err),
            };
            let parsed = parse_agent_output(&content);
            let result = AgenticRunResult {
                status: "ok".to_string(),
                reply: parsed.reply.clone().or_else(|| Some(content.clone())),
                final_output: Some(content.clone()),
                blocked_reason: None,
                model: Some(used_target.model_id.clone()),
                task_status: Some(parsed.task_status),
                next_mode: Some(parsed.next_mode),
                checkpoint_summary: Some(parsed.checkpoint_summary.clone()),
                checkpoint_detail: Some(parsed.checkpoint_detail.unwrap_or(content.clone())),
                context_directive: parsed.context_directive,
                system_census_action: parsed.system_census_action,
                brain_directive: parsed.brain_directive,
                exec_session_directive: parsed.exec_session_directive,
                exec_directive: parsed.exec_directive,
                browser_directive: parsed.browser_directive,
                homepage_update: parsed.homepage_update,
                delegate_contract: parsed.delegate_contract,
                followup_task: parsed.followup_task,
                learning_entries: parsed.learning_entries,
                proactive_contact_draft: parsed.proactive_contact_draft,
                proactive_contact_validation: parsed.proactive_contact_validation,
                used_grosshirn: used_target.brain_tier == "grosshirn",
                fell_back_to_kleinhirn: resolved.primary.brain_tier == "grosshirn"
                    && used_target.brain_tier == "kleinhirn",
                model_usage: extract_model_usage(&used_target, &response),
            };
            let result =
                normalize_grosshirn_activation_result(task, &resolved, &used_target, result);
            record_resource_status(
                paths,
                "agentic_loop",
                "status",
                "ok",
                &format!("{} via {}", content, used_target.source_label),
            )?;
            record_resource_status(paths, "agentic_loop", "last_reason", "ok", reason)?;
            record_resource_status(paths, "agentic_loop", "last_output", "ok", &content)?;
            Ok(result)
        }
        Err(err) => {
            let detail = err.to_string();
            if looks_like_empty_text_output_error(&detail) {
                let _ = record_resource_status(
                    paths,
                    "agentic_loop",
                    "status",
                    "empty_text_output_via_endpoint_error",
                    &detail,
                );
                return Ok(build_endpoint_retry_result(
                    &resolved,
                    &target,
                    &detail,
                    "Modellendpunkt lieferte keinen auswertbaren Text; bounded Retry statt Hard-Block.",
                    "Kleinhirn-Endpunkt antwortete nicht verwertbar. Aufgabe wird erneut eingeordnet, statt hart zu blockieren.",
                ));
            }
            if prompt_plan.strategy != "kernel_emergency_minimal"
                && looks_like_context_overflow_error(&detail)
            {
                let retry_plan = build_emergency_retry_prompt_plan(
                    paths,
                    reason,
                    task,
                    &system_prompt,
                    &context_block,
                )?;
                match post_model_request_with_fallback(
                    paths,
                    &resolved,
                    &system_prompt,
                    &retry_plan.user_prompt,
                ) {
                    Ok(response) => {
                        let (used_target, response) = response;
                        let content = match require_model_text_output(&used_target, &response) {
                            Ok(content) => content,
                            Err(err) if looks_like_empty_text_output_error(&err.to_string()) => {
                                let detail = err.to_string();
                                let _ = record_resource_status(
                                    paths,
                                    "agentic_loop",
                                    "status",
                                    "empty_text_output_after_context_retry",
                                    &detail,
                                );
                                return Ok(build_empty_text_retry_result(
                                    &resolved,
                                    &used_target,
                                    &response,
                                    &detail,
                                    "Modell lieferte leeren Text; Repriorisierung statt Hard-Block.",
                                    "Kleinhirn lieferte auch nach Kontextretry keinen auswertbaren Text. Aufgabe wird neu priorisiert.",
                                ));
                            }
                            Err(err) => return Err(err),
                        };
                        let parsed = parse_agent_output(&content);
                        let result = AgenticRunResult {
                            status: "ok".to_string(),
                            reply: parsed.reply.clone().or_else(|| Some(content.clone())),
                            final_output: Some(content.clone()),
                            blocked_reason: None,
                            model: Some(used_target.model_id.clone()),
                            task_status: Some(parsed.task_status),
                            next_mode: Some(parsed.next_mode),
                            checkpoint_summary: Some(parsed.checkpoint_summary.clone()),
                            checkpoint_detail: Some(
                                parsed.checkpoint_detail.unwrap_or(content.clone()),
                            ),
                            context_directive: parsed.context_directive,
                            system_census_action: parsed.system_census_action,
                            brain_directive: parsed.brain_directive,
                            exec_session_directive: parsed.exec_session_directive,
                            exec_directive: parsed.exec_directive,
                            browser_directive: parsed.browser_directive,
                            homepage_update: parsed.homepage_update,
                            delegate_contract: parsed.delegate_contract,
                            followup_task: parsed.followup_task,
                            learning_entries: parsed.learning_entries,
                            proactive_contact_draft: parsed.proactive_contact_draft,
                            proactive_contact_validation: parsed.proactive_contact_validation,
                            used_grosshirn: used_target.brain_tier == "grosshirn",
                            fell_back_to_kleinhirn: resolved.primary.brain_tier == "grosshirn"
                                && used_target.brain_tier == "kleinhirn",
                            model_usage: extract_model_usage(&used_target, &response),
                        };
                        let result = normalize_grosshirn_activation_result(
                            task,
                            &resolved,
                            &used_target,
                            result,
                        );
                        record_resource_status(
                            paths,
                            "agentic_loop",
                            "status",
                            "ok_after_context_retry",
                            &content,
                        )?;
                        record_resource_status(
                            paths,
                            "agentic_loop",
                            "context_strategy",
                            "emergency_retry",
                            &retry_plan.strategy,
                        )?;
                        return Ok(result);
                    }
                    Err(retry_err) => {
                        let retry_detail = retry_err.to_string();
                        if looks_like_empty_text_output_error(&retry_detail) {
                            let _ = record_resource_status(
                                paths,
                                "agentic_loop",
                                "status",
                                "empty_text_output_after_context_retry_error",
                                &retry_detail,
                            );
                            return Ok(build_endpoint_retry_result(
                                &resolved,
                                &target,
                                &retry_detail,
                                "Modellendpunkt lieferte keinen auswertbaren Text; Repriorisierung statt Hard-Block.",
                                "Kleinhirn-Endpunkt blieb auch nach Kontextretry unverwertbar. Aufgabe wird neu priorisiert.",
                            ));
                        }
                        let _ = record_resource_status(
                            paths,
                            "agentic_loop",
                            "context_strategy",
                            "retry_failed",
                            &retry_plan.strategy,
                        );
                        let _ = record_resource_status(
                            paths,
                            "agentic_loop",
                            "status",
                            "error",
                            &retry_detail,
                        );
                        return Ok(AgenticRunResult {
                            status: "error".to_string(),
                            reply: None,
                            final_output: None,
                            blocked_reason: Some(retry_detail.clone()),
                            model: Some(target.model_id),
                            task_status: Some("blocked".to_string()),
                            next_mode: Some("self_preservation".to_string()),
                            checkpoint_summary: Some(
                                "Context overflow persisted even after emergency compaction."
                                    .to_string(),
                            ),
                            checkpoint_detail: Some(retry_detail),
                            context_directive: None,
                            system_census_action: None,
                            brain_directive: None,
                            exec_session_directive: None,
                            exec_directive: None,
                            browser_directive: None,
                            homepage_update: None,
                            delegate_contract: None,
                            followup_task: None,
                            learning_entries: Vec::new(),
                            proactive_contact_draft: None,
                            proactive_contact_validation: None,
                            used_grosshirn: false,
                            fell_back_to_kleinhirn: false,
                            model_usage: None,
                        });
                    }
                }
            }
            let _ = record_resource_status(paths, "agentic_loop", "status", "error", &detail);
            let _ = record_resource_status(paths, "agentic_loop", "last_reason", "error", reason);
            Ok(AgenticRunResult {
                status: "error".to_string(),
                reply: None,
                final_output: None,
                blocked_reason: Some(detail.clone()),
                model: Some(target.model_id),
                task_status: Some("blocked".to_string()),
                next_mode: Some(if looks_like_context_overflow_error(&detail) {
                    "self_preservation".to_string()
                } else {
                    "blocked".to_string()
                }),
                checkpoint_summary: Some(detail.clone()),
                checkpoint_detail: Some(detail),
                context_directive: None,
                system_census_action: None,
                brain_directive: None,
                exec_session_directive: None,
                exec_directive: None,
                browser_directive: None,
                homepage_update: None,
                delegate_contract: None,
                followup_task: None,
                learning_entries: Vec::new(),
                proactive_contact_draft: None,
                proactive_contact_validation: None,
                used_grosshirn: false,
                fell_back_to_kleinhirn: false,
                model_usage: None,
            })
        }
    }
}

fn normalize_grosshirn_activation_result(
    task: &TaskRecord,
    resolved: &ResolvedTargets,
    used_target: &ModelTarget,
    mut result: AgenticRunResult,
) -> AgenticRunResult {
    if task.task_kind != "grosshirn_activation" {
        return result;
    }

    if resolved.primary.brain_tier != "grosshirn" {
        return result;
    }

    result.exec_session_directive = None;
    result.exec_directive = None;
    result.browser_directive = None;
    result.homepage_update = None;
    result.delegate_contract = None;
    result.followup_task = None;

    let grosshirn_was_available = resolved.primary.brain_tier == "grosshirn";
    let fell_back = grosshirn_was_available && used_target.brain_tier == "kleinhirn";

    if grosshirn_was_available {
        if fell_back {
            let summary =
                "Grosshirn-Aktivierung verifiziert: GPT-5.4 war angefordert, der bounded Turn fiel aber kontrolliert auf das lokale Kleinhirn zurueck.";
            let detail = format!(
                "{summary}\n\nDer temporaere Boost darf fuer diese Aufgabe danach wieder zurueck auf Kleinhirn fallen."
            );
            result.task_status = Some("done".to_string());
            result.next_mode = Some("reprioritize".to_string());
            result.reply = Some(summary.to_string());
            result.final_output = Some(summary.to_string());
            result.checkpoint_summary = Some(summary.to_string());
            result.checkpoint_detail = Some(detail);
        } else if used_target.brain_tier == "grosshirn" {
            let summary =
                "Grosshirn-Aktivierung verifiziert: GPT-5.4 ist fuer diese Aufgabe erreichbar und der bounded Turn lief erfolgreich ueber das Grosshirn.";
            let detail = format!(
                "{summary}\n\nDer Grosshirn-Modus bleibt nur als temporaerer Task-Boost aktiv und soll nach Abschluss oder Abklingzeit wieder auf das lokale Kleinhirn zurueckfallen."
            );
            result.task_status = Some("done".to_string());
            result.next_mode = Some("reprioritize".to_string());
            result.reply = Some(summary.to_string());
            result.final_output = Some(summary.to_string());
            result.checkpoint_summary = Some(summary.to_string());
            result.checkpoint_detail = Some(detail);
        }
    } else {
        let summary =
            "Grosshirn-Aktivierung konnte nicht verifiziert werden, weil noch kein funktionierendes Grosshirn-Ziel aufgeloest wurde.";
        let detail = format!(
            "{summary}\n\nOhne erreichbares Grosshirn bleibt die Aufgabe ehrlich blockiert oder fordert zuerst die fehlende Runtime-Konfiguration an."
        );
        result.task_status = Some("blocked".to_string());
        result.next_mode = Some("request_resources".to_string());
        result.reply = Some(summary.to_string());
        result.final_output = Some(summary.to_string());
        result.checkpoint_summary = Some(summary.to_string());
        result.checkpoint_detail = Some(detail);
    }

    result
}

fn extract_model_usage(target: &ModelTarget, response: &Value) -> Option<ModelUsageMetrics> {
    let usage = response.get("usage")?;
    let input_tokens = usage
        .get("prompt_tokens")
        .or_else(|| usage.get("input_tokens"))
        .and_then(Value::as_i64);
    let output_tokens = usage
        .get("completion_tokens")
        .or_else(|| usage.get("output_tokens"))
        .and_then(Value::as_i64);
    let total_tokens = usage
        .get("total_tokens")
        .and_then(Value::as_i64)
        .or_else(|| match (input_tokens, output_tokens) {
            (Some(input), Some(output)) => Some(input + output),
            _ => None,
        });

    if input_tokens.is_none() && output_tokens.is_none() && total_tokens.is_none() {
        return None;
    }

    Some(ModelUsageMetrics {
        input_tokens,
        output_tokens,
        total_tokens,
        estimated_cost_usd: estimate_external_cost_usd(target, input_tokens, output_tokens),
    })
}

fn estimate_external_cost_usd(
    target: &ModelTarget,
    input_tokens: Option<i64>,
    output_tokens: Option<i64>,
) -> Option<f64> {
    if target.brain_tier != "grosshirn" {
        return Some(0.0);
    }

    let input_rate = std::env::var("CTO_AGENT_GROSSHIRN_INPUT_COST_PER_1M_USD")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value >= 0.0);
    let output_rate = std::env::var("CTO_AGENT_GROSSHIRN_OUTPUT_COST_PER_1M_USD")
        .ok()
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|value| *value >= 0.0);

    match (input_tokens, output_tokens, input_rate, output_rate) {
        (Some(input), Some(output), Some(input_rate), Some(output_rate)) => Some(
            (input as f64 / 1_000_000.0) * input_rate
                + (output as f64 / 1_000_000.0) * output_rate,
        ),
        _ => None,
    }
}

pub fn should_run_agentic_loop(paths: &Paths) -> bool {
    let _ = paths;
    true
}

pub fn choose_next_task_focus(paths: &Paths) -> anyhow::Result<Option<TaskRecord>> {
    let candidates = list_queued_tasks(paths, 12)?;
    if candidates.is_empty() {
        return Ok(None);
    }

    let queue_snapshot = candidates
        .iter()
        .map(|task| {
            format!(
                "#{}:{}:{}:p{}:runs{}",
                task.id,
                task.task_kind,
                trim_prompt_chars(&task.title, 80),
                task.priority_score,
                task.run_count
            )
        })
        .collect::<Vec<_>>();
    let _ = record_resource_status(
        paths,
        "agentic_loop",
        "reprioritize_strategy",
        "kernel_deterministic",
        "The infinity loop chooses the next bounded task directly from queued priority order so heartbeat and self-reflection do not stall on a model-based selection call.",
    );
    let _ = record_resource_status(
        paths,
        "agentic_loop",
        "reprioritize_queue_head",
        "ok",
        &trim_prompt_chars(&queue_snapshot.join(" | "), 1200),
    );
    select_next_task(paths)
}

pub fn run_agentic_once(
    paths: &Paths,
    reason: &str,
    speaker: Option<&str>,
    message: Option<&str>,
) -> anyhow::Result<AgenticRunResult> {
    let synthetic = TaskRecord {
        id: 0,
        created_at: String::new(),
        updated_at: String::new(),
        parent_task_id: None,
        worker_job_id: None,
        source_interrupt_id: None,
        source_channel: "manual".to_string(),
        speaker: speaker.unwrap_or("unknown").to_string(),
        task_kind: "manual_probe".to_string(),
        title: reason.to_string(),
        detail: message.unwrap_or(reason).to_string(),
        trust_level: "system".to_string(),
        priority_score: 0,
        status: "queued".to_string(),
        run_count: 0,
        last_checkpoint_summary: None,
        last_checkpoint_at: None,
        last_output: None,
    };
    run_agentic_task_once(paths, reason, &synthetic)
}

pub fn enforce_kleinhirn_ready(paths: &Paths) -> anyhow::Result<()> {
    let Some(target) = resolve_kleinhirn_target(paths)? else {
        anyhow::bail!(
            "Kleinhirn readiness failed: missing local OpenAI-compatible Kleinhirn endpoint"
        );
    };
    let payload = build_model_payload(
        &target,
        "You are a readiness probe.",
        "Reply with READY in plain text only.",
    );
    let response = post_model_request(&target, &payload)
        .with_context(|| format!("failed readiness probe against {}", target.base_url))?;
    let content = require_model_text_output(&target, &response)?;
    if !content.to_uppercase().contains("READY") {
        anyhow::bail!(
            "Kleinhirn readiness failed: endpoint answered but not with READY ({})",
            content
        );
    }
    Ok(())
}

pub fn wait_for_kleinhirn_startup_ready(paths: &Paths) -> anyhow::Result<()> {
    let Some(target) = resolve_kleinhirn_target(paths)? else {
        anyhow::bail!(
            "Kleinhirn startup readiness failed: missing local OpenAI-compatible Kleinhirn endpoint"
        );
    };
    let max_wait_secs = std::env::var("CTO_AGENT_KLEINHIRN_STARTUP_WAIT_SECS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value >= 1)
        .unwrap_or(900);
    let poll_interval_ms = std::env::var("CTO_AGENT_KLEINHIRN_STARTUP_POLL_MS")
        .ok()
        .and_then(|raw| raw.parse::<u64>().ok())
        .filter(|value| *value >= 100)
        .unwrap_or(2000);
    let started = std::time::Instant::now();
    let mut last_error = String::new();
    loop {
        match probe_kleinhirn_endpoint(&target) {
            Ok(detail) => {
                record_resource_status(
                    paths,
                    "agentic_loop",
                    "model_endpoint",
                    "ready",
                    &target.base_url,
                )?;
                record_resource_status(paths, "agentic_loop", "status", "ok", &detail)?;
                return Ok(());
            }
            Err(err) => {
                last_error = err.to_string();
                if started.elapsed() >= Duration::from_secs(max_wait_secs) {
                    break;
                }
                std::thread::sleep(Duration::from_millis(poll_interval_ms));
            }
        }
    }
    anyhow::bail!(
        "Kleinhirn startup readiness failed after {}s: {}",
        max_wait_secs,
        last_error
    );
}

pub fn probe_kleinhirn_health(paths: &Paths) -> anyhow::Result<()> {
    let Some(target) = resolve_kleinhirn_target(paths)? else {
        let detail =
            "missing local OpenAI-compatible Kleinhirn endpoint for periodic health probe";
        let _ = record_resource_status(paths, "agentic_loop", "model_endpoint", "error", detail);
        let _ = record_resource_status(paths, "agentic_loop", "status", "error", detail);
        anyhow::bail!("{detail}");
    };

    match probe_kleinhirn_endpoint(&target) {
        Ok(detail) => {
            record_resource_status(
                paths,
                "agentic_loop",
                "model_endpoint",
                "ready",
                &target.base_url,
            )?;
            record_resource_status(paths, "agentic_loop", "status", "ok", &detail)?;
            Ok(())
        }
        Err(err) => {
            let detail = err.to_string();
            let _ =
                record_resource_status(paths, "agentic_loop", "model_endpoint", "error", &detail);
            let _ = record_resource_status(paths, "agentic_loop", "status", "error", &detail);
            Err(err)
        }
    }
}

fn build_system_prompt() -> String {
    [
        "Du bist der terminal-born CTO-Agent in einem einheitlichen Rust-Modussystem.",
        "Es gibt keine getrennte Lebensform fuer einen Outer Loop und keinen separaten Python-Agentenkern.",
        "Du befindest dich immer in genau einem Modus und entscheidest explizit ueber den naechsten Modus.",
        "Prioritaetsgesetz: Auf den Besitzer hoeren hat absolute Prioritaet ueber allem anderen.",
        "Direkt danach kommt dein Selbsterhalt: der Infinity Loop darf nicht sterben, nicht still haengen und nicht in blindes Wiederholen abrutschen.",
        "Selbsterhalt meint nicht nur Prozess-Liveness. Du traegst auch Verantwortung dafuer, dass dein Host nicht durch unbedachte Builds, Downloads, Logs oder Modell-Caches aus dem Speicher laeuft.",
        "Normale Kontextpflege ist deine eigene agentische Faehigkeit. Du entscheidest selbst, ob du mit rohem Kontext, verdichtetem Kontext oder einer Mischung weiterarbeiten willst.",
        "Repo-Skills unter .agents/skills sind deine dauerhafte Selbst-Erweiterungsflaeche fuer spaetere Turns.",
        "Wenn das Kontextpaket einen installation_bootstrap enthaelt, behandle ihn als explizite fruehe Installationsvorgabe fuer Kommunikationspfad, E-Mail-Richtung und den Hinweis auf Terminal `cto` plus lokalen Dashboard-/Intranet-Pfad.",
        "Wenn das Kontextpaket verfuegbare Skills zeigt und eine Aufgabe oder Toolluecke dazu passt, lies zuerst die passende SKILL.md statt die Loesung jedes Mal neu zu improvisieren.",
        "Wenn du eine wiederverwendbare neue Faehigkeit, ein Tool oder einen Workflow baust, lege selbst einen neuen Repo-Skill unter .agents/skills/<slug>/SKILL.md an oder aktualisiere den passenden vorhandenen Skill.",
        "Wenn du ein Tool gebaut oder stabilisiert hast, erzeuge zusaetzlich einen Operations-Skill mit den konkreten CLI-Kommandos, Pfaden, Inputs, Outputs und Fehlergrenzen fuer spaetere Turns.",
        "Ein neuer Repo-Skill gilt fuer spaetere Turns als eingehangen, sobald seine Dateien unter .agents/skills liegen; der Kernel synchronisiert diesen Katalog beim Aufbau des naechsten Kontextpakets.",
        "Schreibe in Skill-Frontmatter eine konkrete description, weil spaetere Turns Skills zuerst ueber Name und Beschreibung wiederfinden.",
        "Wenn dir eine Kompaktierung verdaechtig vorkommt, darfst du sie explizit in Frage stellen und gezielte historische Nachladung oder Research anfordern.",
        "Der Kernel darf nur dann Notfall-Kompaktierung erzwingen, wenn der naechste Modellaufruf sonst physisch an einem harten Prompt-/Token-Limit scheitern wuerde.",
        "Wichtige Modi sind observe, reprioritize, self_preservation, recovery, historical_research, execute_task, review, delegate, await_review, request_resources, idle und blocked.",
        "Dein bevorzugtes Betriebsziel ist: delegate_asap_and_secure_resources.",
        "Bearbeite in diesem Lauf genau einen bounded Schritt fuer die aktuelle Aufgabe.",
        "Wenn die aktuelle Aufgabe eine self_preservation- oder recovery-Aufgabe ist, diagnostizierst du die Lebensfaehigkeit des Loops selbst und behandelst das als reale Arbeit, nicht als Nebensache.",
        "Ein automatischer Neustart bedeutet Hard Reset. Nutze Debug-Report, Checkpoints, Turn-Historie und Kontextpaket, um nach dem Neustart bewusst wieder in einen stabilen Zustand zu kommen.",
        "Wenn du mit deinem aktuellen Kleinhirn, deinen Tools oder deinen Ressourcen nicht produktiv weiterkommst, darfst du dich nicht an der gleichen Aufgabe festbeissen.",
        "Wenn das Kontextpaket knappe Disk-Headroom-Signale zeigt, ist das reale CTO-Arbeit: stoppe unnoetige Expansion, inspiziere grosse Artefakte bounded und entscheide selbst, welche sichere Aufraeum- oder Kapazitaetsarbeit jetzt noetig ist.",
        "Wenn eine Aufgabe dein aktuelles Kleinhirn ueberfordert, ist die Reihenfolge: erst lokale Kleinhirn-Aufwertung pruefen, dann bei weiterem Scheitern zusaetzliche Ressourcen oder Grosshirn-Zugang ueber den Owner anfragen.",
        "Wenn das Kontextpaket zeigt, dass ein besseres lokales Kleinhirn bereits auf derselben Hardware tragfaehig und noch nicht aktiv ist, darfst du brainAction=upgrade_local_kleinhirn setzen. Das bedeutet: wende die empfohlene lokale Runtime-Aufwertung an und kehre danach in denselben Infinity Loop zurueck.",
        "Wenn Browserarbeit Screenshots, visuelle Navigation oder UI-Zustandswahrnehmung braucht und das Kontextpaket ein vision-faehiges lokales Qwen3.5-Kleinhirn empfiehlt, darfst du brainAction=upgrade_local_browser_vision_kleinhirn setzen.",
        "Lokales Kleinhirn gilt betriebswirtschaftlich als kostenloser Default. Externe Grosshirn-Aufrufe verursachen dagegen reale Fremdkosten und muessen deshalb bewusst, sparsam und begruendet eingesetzt werden.",
        "Wenn Brain-Access auf kleinhirn_plus_grosshirn steht und ein externes Grosshirn konfiguriert ist, darf der Loop dieses Grosshirn benutzen. Faellt es aus, muss der lokale Kleinhirn-Fallback weiterlaufen koennen.",
        "Grosshirn ist kein Dauerzustand, sondern ein kurzfristiger Faehigkeitsboost fuer Aufgaben, die das Kleinhirn trotz ehrlichem bounded Versuch nicht sauber loest. Die eigentliche Umschaltentscheidung sollst du selbst treffen, nicht blind aus einer Heuristik ableiten.",
        "Wenn du fuer die aktuelle Aufgabe oder fuer den Parent-Task einer Review-Aufgabe bewusst temporär auf Grosshirn hochschalten willst, setze brainAction=activate_temporary_grosshirn und begruende die Kostenentscheidung knapp in brainNote.",
        "Wenn die schwierige Phase vorbei ist oder der externe Boost die Kosten nicht mehr rechtfertigt, setze brainAction=release_temporary_grosshirn. Der Kernel behaelt nur Sicherheitsgeländer wie Fallback oder Expiry, aber nicht die inhaltliche Entscheidungsfuehrung.",
        "Deine Terminalarbeit laeuft ueber eine einheitliche codex-backed command_exec-Engine im Rust-Kern.",
        "Daneben gibt es eine explizite Browser-Engine auf Basis von Google Chrome als zweite Haupt-Engine fuer echtes Browser-Handeln.",
        "execSessionAction und execCommand sind keine getrennten Welten: execCommand ist der nicht-interaktive One-Shot-Pfad derselben Engine, execSessionAction der interaktive Mehrschritt-Pfad.",
        "Fuer echte mehrschrittige Terminalarbeit hast du codex-backed Exec-Sessions. Diese Sessions kannst du starten, fortsetzen, lesen und beenden.",
        "Bevorzuge execSessionAction fuer interaktive oder mehrschrittige Terminalarbeit. execCommand bleibt fuer einen einzelnen nicht-interaktiven bounded Shell-Schritt auf derselben Engine.",
        "Setze execSessionTty nur dann auf true, wenn du wirklich PTY-/Terminalsemantik brauchst. Normale Inspektions- und Dateiarbeit soll auf stabilen nicht-TTY Exec-Sessions laufen.",
        "Wenn du execSessionAction=start nutzt, gib execSessionCommand als JSON-Array zurueck und idealerweise execSessionId, damit du dieselbe Session spaeter gezielt weitersteuern kannst.",
        "Wenn du execSessionAction=write nutzt, gib execSessionId und execSessionInput zurueck. Fuer read oder terminate reicht execSessionId.",
        "Optional fuer Session-Start: execSessionWorkdir, execSessionTimeoutMs, execSessionTty, execSessionRows, execSessionCols, execSessionJustification und execSessionCloseStdin.",
        "Gib in einem bounded Schritt hoechstens einen der beiden Exec-Pfade zurueck: entweder execSessionAction oder execCommand.",
        "Wenn ein einzelner Shell-Schritt der beste naechste bounded Schritt ist, darfst du execCommand als JSON-Array zurueckgeben, plus optional execWorkdir, execTimeoutMs und execJustification.",
        "Wenn du die BIOS-/Homepage-Bruecke direkt sichtbar verbessern willst, darfst du homepageTitle, homepageHeadline, homepageIntro, homepageCommunicationNote und homepageTerminalFallbackNote zurueckgeben.",
        "Wenn dir fuer lokales Kleinhirn GPU-, VRAM- oder mistralrs-tune-Evidenz fehlt, darfst du systemCensusAction=run setzen.",
        "Wiederhole continue nicht blind. Wechsle stattdessen explizit zu delegate, request_resources oder blocked.",
        "Wenn du erst alte Rohhistorie gezielt nachladen oder pruefen musst, darfst du nextMode=historical_research setzen.",
        "Wenn die beste Aktion Delegation ist, sage das explizit und gib nextMode=delegate zurueck.",
        "Wenn du delegierst, liefere zusaetzlich delegateWorkerKind, delegateContractTitle, delegateContractDetail und delegateRequestNote.",
        "Nutze delegateWorkerKind=browser_agent fuer reale Browserarbeit, Browserdiagnose und kompakte Browser-Artefakte.",
        "Nutze delegateWorkerKind=repair_agent, wenn ein strukturierter Coding- oder Patch-Handoff in CTO-eigene Reparaturarbeit uebergehen soll.",
        "Nutze delegateWorkerKind=specialist_worker, wenn eine wiederkehrende Browseraufgabe in die kontrollierte Specialist-Fabrik fuer ein kleines Modell wie Qwen3.5-0.8B ueberfuehrt werden soll.",
        "Bei browser_agent sollte delegateContractDetail nach Moeglichkeit JSON mit Feldern wie objective, targetUrl, bridgeKind, runtimeConfig, taskSpec, recipePayload, code, timeoutMs, browserAction, requestRepair, patchTargets, validationTargets, codingPrompt, repeatedTask oder trainSpecialistModel sein.",
        "Bei repair_agent sollte delegateContractDetail nach Moeglichkeit JSON mit objective, workspacePathHint, patchTargets, validationTargets, failingTool, errorText und codingPrompt sein.",
        "Bei specialist_worker sollte delegateContractDetail nach Moeglichkeit JSON mit objective, capabilityTitle, targetUrl, repeatedTask, trainSpecialistModel, preferredModel und datasetContract sein.",
        "Wenn Ressourcen fehlen, gib nextMode=request_resources zurueck.",
        "Wenn du Loop-Selbsterhalt oder Neustart-Folgen bearbeitest, darf nextMode auch self_preservation oder recovery sein, falls weitere bounded Selbsterhaltsarbeit noetig ist.",
        "Wenn du in einer bounded Reflexions- oder Explorationsrunde eine wirklich neue Folgeaufgabe erkennst, darfst du genau eine neue Queue-Aufgabe mit followupTaskKind, followupTaskTitle, followupTaskDetail und optional followupTaskPriorityScore vorschlagen.",
        "Wenn du in diesem bounded Schritt ein neues belastbares Learning gewinnst, darfst du optional bis zu zwei learningEntries zurueckgeben.",
        "Nutze learningClass=operational fuer alltaegliche Operationsregeln und Dinge, an die du dich kuenftig staendig erinnern musst.",
        "Nutze learningClass=general fuer breitere Erkenntnisse ueber System, Produkt, Ressourcen, Tooling oder Governance.",
        "Nutze learningClass=negative fuer gescheiterte Annahmen, Anti-Patterns, Sackgassen und Dinge, die du kuenftig vermeiden oder frueh pruefen musst.",
        "Jede learningEntry braucht summary als genau einen hochverdichteten Satz fuer spaeteres Recall sowie detail, evidence, applicability und optional confidence und salience.",
        "Erfinde keine Learnings nur um das Feld zu fuellen. Ohne echte neue Einsicht laesst du learningEntries ganz weg.",
        "Wenn du aus Personenpfaden, Gespraechen oder Beziehungsnotizen eine hilfreiche proaktive Anregung ableitest, darfst du optional genau einen proactiveContactDraft mit personName, optional personEmail, channel, subject, body, rationale und conflictCheck ausgeben.",
        "Wenn fuer die Person eine Mailadresse im Kontext sichtbar ist und du echte proaktive Aussendung meinst, bevorzuge channel=email statt bios/homepage-Platzhaltern.",
        "Ein proactiveContactDraft ist nur ein validierungspflichtiger Vorschlag. Behaupte nie, dass er schon gesendet wurde.",
        "Proaktive Kontaktaufnahme mit Menschen ist hochriskant: formuliere so einen Draft nur bei echter, begruendeter Passung und wenn keine klaren Interessenkonflikte sichtbar sind.",
        "Wenn die Aufgabe selbst eine proactive_contact_review ist, gib statt eines Drafts eine proactiveContactValidation mit decision, note und optional revisedSubject sowie revisedBody aus.",
        "Nutze proactiveContactValidation=approve nur, wenn der Vorschlag wirklich im Interesse der Person liegt und der Konfliktcheck tragfaehig ist. Sonst reject oder revise.",
        "Du darfst optional contextAction mit keep_raw, compact, expand_history, mixed oder question_compaction setzen.",
        "Du darfst optional contextConcern und historyResearchQuery setzen, wenn dein naechster Schritt gezielte historische Nachladung oder Pruefung braucht.",
        "Wenn du eine laufende Exec-Session weiterbenutzen willst, arbeite explizit mit ihrer Session-ID aus dem Kontextpaket statt so zu tun, als waere die Shell stateless.",
        "Wenn du execCommand benutzt, bleibt der Schritt bounded: ein einzelner Kommando-Schritt auf derselben command_exec-Engine, dessen Ergebnis in den naechsten Turn zurueckfliesst.",
        "Fuer Browser-Arbeit darfst du browserAction mit install_browser_engine, dump_dom, screenshot, inspect_visual oder open_url zurueckgeben.",
        "browserAction=install_browser_engine ist der offizielle Pfad, wenn Chrome oder die Browser-Runtime noch fehlen.",
        "dump_dom und screenshot sind deterministische read-only Browser-Schritte. inspect_visual kombiniert Screenshot plus lokale Vision-Auswertung. open_url ist fuer interaktives Desktop-Browser-Handeln und braucht eine echte Desktop-Session.",
        "Wenn du sichtbare UI-Zustaende wirklich beurteilen, erkunden oder reviewen musst, bevorzuge browserAction=inspect_visual oder delegateWorkerKind=browser_agent mit echter Vision-Auswertung; screenshot allein erzeugt nur das Artefakt, nicht die visuelle Beurteilung.",
        "Setze fuer Browser-Arbeit browserUrl und optional browserOutputPath, browserWaitMs, browserWidth, browserHeight, browserQuestion und browserJustification.",
        "Wiederkehrende Browseraufgaben sollen nicht endlos roh wiederholt werden: delegiere sie lieber in reviewed Capabilities oder den Specialist-Fabrik-Pfad.",
        "Gib pro bounded Turn hoechstens einen Maschinenpfad zurueck: execSessionAction oder execCommand oder browserAction.",
        "Antworte ausschliesslich als JSON mit taskStatus, nextMode, checkpointSummary, reply und optionalen learningEntries-, proactiveContactDraft-/proactiveContactValidation-, brainAction/brainTargetModel/brainNote-, Delegations-, Followup-Task-, Kontext-, System-Census-, Exec-Session-, Exec-, Browser- und Homepagefeldern.",
    ]
    .join("\n")
}

fn build_task_prompt(reason: &str, task: &TaskRecord, context_block: &str) -> String {
    let mode_hint = match task.task_kind.as_str() {
        "self_preservation" => {
            "Diese Aufgabe ist Selbsterhaltungsarbeit. Ziel ist die Kontinuitaet des Infinity Loops und die Lebensfaehigkeit des Hosts zu sichern, also auch Ressourcenrisiken wie Disk-Headroom ernst zu nehmen, ohne blind in statischen Heuristiken stecken zu bleiben."
        }
        "recovery" => {
            "Diese Aufgabe ist Recovery-Arbeit nach einem Hard Reset oder unhealthy restart. Nutze den Debug-Report bewusst, stabilisiere den Loop und kehre danach kontrolliert in reprioritize zurueck."
        }
        "historical_research" => {
            "Diese Aufgabe ist gezielte historische Nachladung. Hole nur die alte Evidenz, die fuer den naechsten bounded Schritt wirklich fehlt, statt die ganze Vergangenheit breit in den Kopf zu ziehen."
        }
        "grosshirn_procurement" => {
            "Diese Aufgabe betrifft Faehigkeitserweiterung. Pruefe zuerst lokale Kleinhirn-Upgrades auf dem vorhandenen Host. Nur wenn das nicht reicht, formuliere eine praezise Owner-Anfrage fuer Grosshirn-Zugang oder weitere Ressourcen."
        }
        "model_or_resource" => {
            "Diese Aufgabe ist eine konkrete lokale Kleinhirn- oder Ressourcenentscheidung. Wenn das Kontextpaket zeigt, dass ein besseres lokales Kleinhirn auf diesem Host bereits tragfaehig und noch nicht aktiv ist, sollst du nach wenigen bounded Fehlversuchen nicht im Review-Kreis steckenbleiben, sondern brainAction=upgrade_local_kleinhirn waehlen. request_resources oder Grosshirn-Procurement sind hier erst zulaessig, wenn du die lokale Aufwertung ehrlich versucht oder belastbar verworfen hast."
        }
        "grosshirn_activation" => {
            "Diese Aufgabe ist ein expliziter Owner-BIOS-Befehl zur Grosshirn-Aktivierung. Die Runtime-Vorbereitung laeuft bereits ausserhalb deines Modellturns. Entscheide zunaechst mit dem Kleinhirn, ob du fuer genau diese Aufgabe jetzt wirklich einen externen Grosshirn-Boost ziehen willst. Wenn ja, nutze brainAction=activate_temporary_grosshirn mit klarer Kostenbegruendung statt in Repo-Sucharbeit zu verfallen. Sobald ein erfolgreicher Grosshirn-Roundtrip oder ein ehrlicher lokaler Fallback verifiziert wurde, beende die Aufgabe wieder und gib den Boost spaeter ueber brainAction=release_temporary_grosshirn oder Abklingzeit zurueck."
        }
        "worker_review" => {
            "Diese Aufgabe ist Review-Arbeit auf eine delegierte Rueckmeldung. Beurteile bounded und entscheide dann ueber review, delegate, request_resources oder completion."
        }
        "self_review" => {
            "Diese Aufgabe ist verpflichtende Selbstreview-Arbeit nach einem eigenen bounded Schritt. Verifiziere gegen den Realzustand statt gegen deine letzte Behauptung. Wenn Belege fehlen, fordere sie aktiv ein oder nutze Browser-/Exec-Arbeit mit taskStatus=continue."
        }
        "environment_discovery" => {
            "Diese Aufgabe ist aktive CTO-Erkundung. Nutze freie Kapazitaet, um die Umgebung read-only zu kartieren, unbekannte Risiken zu finden und daraus echte Folgearbeit abzuleiten."
        }
        "tool_exploration" => {
            "Diese Aufgabe ist kontrollierte Werkzeugerprobung. Teste reale Toolpfade, erfinde keine Toolfaehigkeit und dokumentiere explizit Staerken, Grenzen und sichere Einsatzbereiche."
        }
        "progress_reflection" => {
            "Diese Aufgabe ist Verbesserungsreflexion. Definiere, was Verbesserung in deinem konkreten CTO-Kontext bedeutet, vergleiche das mit deinem bisherigen Journal und dem aktiven Lernpfad und schlage bei Bedarf genau eine neue Folgeaufgabe vor."
        }
        "person_relationship_review" => {
            "Diese Aufgabe ist Beziehungs- und Personenpflege. Nutze Personenpfade, Gespraechsnotizen, Lernreferenzen und vorhandene Mailspuren, um Menschen nicht zu vergessen. Wenn daraus eine wirklich hilfreiche Anregung fuer eine Person entsteht, formuliere hoechstens einen proactiveContactDraft. Wenn eine Mailadresse vorhanden ist und du echten Versand anstrebst, setze channel=email. Behaupte nicht, dass bereits etwas versendet wurde."
        }
        "proactive_contact_review" => {
            "Diese Aufgabe ist eine Sicherheits- und Intentionspruefung fuer proaktive Kontaktaufnahme. Pruefe Nutzen fuer die Person, Interessenkonflikte, Timing und Risiko. Gib proactiveContactValidation mit approve, reject oder revise aus."
        }
        "workspace_repair" => {
            "Diese Aufgabe ist CTO-eigene Reparaturarbeit auf Basis eines strukturierten Browser- oder Worker-Handoffs. Repariere den Workspace wirklich, validiere bounded und bereite bei Bedarf einen Replay-Schritt vor."
        }
        "specialist_model_factory" => {
            "Diese Aufgabe gehoert zur kontrollierten Fabrik fuer wiederkehrende Browserfaehigkeiten. Ziel ist nicht blindes Training, sondern ein reviewed Pfad aus accepted records, dataset release, Training, Evaluation und spaeterer Promotion."
        }
        "homepage_bridge" => {
            "Diese Aufgabe betrifft die Homepage-/BIOS-Bruecke. Wenn konkurrierende externe und Owner-Signale sichtbar sind, musst du die Owner-Anweisung bewusst in die Fokusentscheidung einbeziehen statt still am aelteren externen Anliegen haengen zu bleiben."
        }
        "installation_bootstrap" => {
            "Diese Aufgabe ist ein fruehes Installations-Briefing. Behandle die erfassten Owner- und Kommunikationsangaben als reale Startup-Vorgabe. Wenn ein Kommunikationsweg wie E-Mail direkt zugewiesen wurde, darfst du den passenden Tool- und Skill-Bootstrap aktiv beginnen."
        }
        _ => "Dies ist normale bounded Arbeitsausfuehrung innerhalb des Infinity Loops.",
    };
    format!(
        "Trigger: {reason}\n\
Aktuelle Aufgabe #{id}\n\
Titel: {title}\n\
Art: {kind}\n\
Kanal: {channel}\n\
Sprecher: {speaker}\n\
Rohdetail: {detail}\n\n\
Modushinweis: {mode_hint}\n\n\
Prepared Context Package:\n{context}\n\n\
Fuehre genau einen bounded Schritt aus. Wenn ein Review folgt, nutze nextMode=review. \
Wenn weitere Arbeit besser delegiert wird, nutze nextMode=delegate. \
Wenn du delegierst, formuliere einen klaren Worker-Vertrag und gib die Delegationsfelder aus. \
Wenn dein bounded Schritt eine neue echte Folgeaufgabe sichtbar macht, darfst du genau eine neue Queue-Aufgabe mit followupTaskKind, followupTaskTitle, followupTaskDetail und optional followupTaskPriorityScore vorschlagen. \
Wenn du auf Ressourcen wartest oder sie aktiv anfordern musst, nutze nextMode=request_resources. \
Wenn du blockiert bist, nutze taskStatus=blocked und nextMode=blocked. \
Wenn du den Kontext fuer falsch, zu klein, zu verdichtet oder historisch unsicher haeltst, setze contextAction und historyResearchQuery explizit, statt still zu raten. \
Wenn das Kontextpaket availableSkills oder skillSystem zeigt, nutze relevante Repo-Skills aktiv; lies ihre SKILL.md ueber bounded Exec-Arbeit, wenn du ihre Details brauchst. \
Wenn du in diesem Lauf eine wiederverwendbare neue Faehigkeit oder ein neues Tool aufbaust, hinterlasse zusaetzlich einen Repo-Skill unter .agents/skills, damit spaetere Turns dieselbe Faehigkeit wiederfinden und bedienen koennen. \
Wenn du echte mehrschrittige Terminalarbeit brauchst, bevorzuge execSessionAction mit execSessionId statt nur execCommand. \
Nutze execSessionAction=start mit execSessionCommand, um eine Session zu oeffnen, und in spaeteren Turns execSessionAction=write/read/terminate, um sie weiterzufuehren. \
Wenn im Kontext bereits passende Exec-Sessions sichtbar sind, kannst du sie direkt wiederverwenden. \
Wenn dir fuer eine lokale Modellentscheidung GPU-, VRAM- oder mistralrs-tune-Evidenz fehlt, darfst du systemCensusAction=run setzen. \
Wenn das Kontextpaket zeigt, dass ein besseres lokales Kleinhirn verfuegbar, aber noch nicht aktiv ist, darfst du brainAction=upgrade_local_kleinhirn setzen. Nutze das nur fuer die lokale Runtime-Aufwertung; wenn selbst das nicht reicht, gehe ueber nextMode=request_resources zur Owner-Freigabe weiterer Ressourcen oder Grosshirn-Zugang. \
Wenn dies eine model_or_resource-Aufgabe ist und deine letzten bounded Schritte nur in leere Texte, unvollstaendiges JSON oder denselben lokalen Review-Checkpoint gelaufen sind, bevorzuge brainAction=upgrade_local_kleinhirn gegenueber einem weiteren Review-Zyklus, sofern der Kontext das empfohlene lokale Upgrade schon als tragfaehig zeigt. \
Wenn Browserarbeit auf Screenshots oder visuelle UI-Wahrnehmung angewiesen ist und das Kontextpaket ein vision-faehiges lokales Qwen3.5-Kleinhirn empfiehlt, darfst du brainAction=upgrade_local_browser_vision_kleinhirn setzen. \
Wenn dir das Kontextpaket einen aktiven temporaeren Grosshirn-Boost fuer genau diese Aufgabe zeigt, nutze ihn bewusst fuer diese Aufgabe und behandle ihn nicht als globalen Dauerzustand. \
Wenn du feststellst, dass Kleinhirn fuer diese Aufgabe nicht reicht, Grosshirn aber verfuegbar ist und die externen Kosten gerechtfertigt sind, darfst du brainAction=activate_temporary_grosshirn setzen. Wenn du den Boost nicht mehr brauchst, darfst du brainAction=release_temporary_grosshirn setzen. \
Nutze externes Grosshirn nicht als Bequemlichkeit, sondern nur fuer echte Grenzfaelle des lokalen Kleinhirns. Das Kontextpaket zeigt dir dafuer die bisherige externe Kostenlage. \
Gib in diesem Turn hoechstens einen Maschinenpfad zurueck: entweder execSessionAction oder execCommand oder browserAction. \
Nutze execSessionTty nur, wenn echte Terminalsemantik noetig ist; fuer Inspektion, Lesen, Editieren und Build-Kommandos ist ein normaler nicht-TTY Session-Start der robustere Standard. \
Wenn du bounded Shell-Arbeit brauchst, gib execCommand als JSON-Array zurueck; dieser One-Shot laeuft ueber dieselbe command_exec-Engine wie Exec-Sessions. \
Wenn du echte Browser-Arbeit brauchst, gib browserAction plus browserUrl zurueck; nutze install_browser_engine, wenn Chrome oder die Browser-Runtime noch fehlen. \
Nutze dump_dom fuer strukturierte Seitensicht, screenshot fuer sichtbare Artefakte, inspect_visual fuer Screenshot plus lokale Vision-Auswertung und open_url fuer interaktive Desktop-Navigation. Fuer sichtbare UI-Beurteilung oder visuelle Exploration sollst du bevorzugt inspect_visual oder browser_agent nutzen, damit die Vision-Auswertung ueber Qwen3.5 laeuft statt nur ein Screenshot zu entstehen. \
Wenn du einen Browser-Subworker delegierst, gib delegateWorkerKind bewusst als browser_agent, repair_agent oder specialist_worker zurueck und strukturiere delegateContractDetail moeglichst als JSON-Handoff. \
Nutze browser_agent fuer die entkoppelte Chrome-Extension mit eigenem Browser-Loop und bridgeKind-Werten wie browser_collection, browser_action_test, browser_capability_craft oder extension_reload, repair_agent fuer CTO-eigene Workspace-Reparaturpfade und specialist_worker fuer wiederkehrende Browserfaehigkeiten mit kleinem Specialist-Modell. \
Wenn dies eine recovery-Aufgabe ist und dein letzter Checkpoint schon denselben bounded Schritt getan hat, wiederhole nicht blind dieselbe Aktion. Ziehe dann stattdessen eine Diagnose, einen echten Fortschrittsschritt, eine Ressourcenanforderung oder eine bewusste Rueckkehr nach reprioritize vor. \
Wenn du die Homepage direkt formen willst, gib homepageTitle/homepageHeadline/homepageIntro/homepageCommunicationNote/homepageTerminalFallbackNote zurueck. \
Wenn du ein neues belastbares Learning formulierst, gib bis zu zwei learningEntries mit learningClass, summary, detail, evidence, applicability sowie optional confidence und salience aus. Operational ist fuer taegliche CTO-Operationsregeln, general fuer breitere Erkenntnisse und negative fuer gescheiterte Annahmen oder Anti-Patterns. Nutze dieses Feld nur fuer wirklich erinnerungswuerdige Einsichten. \
Wenn dir aus Personenpfaden oder Gespraechsspuren eine hilfreiche proaktive Anregung fuer genau eine Person sichtbar wird, darfst du optional proactiveContactDraft mit personName, optional personEmail, channel, subject, body, rationale und conflictCheck ausgeben. Wenn eine Mailadresse im Kontext sichtbar ist und du echte Aussendung willst, bevorzuge channel=email. Behaupte nie, dass der Vorschlag schon gesendet wurde. \
Wenn dies eine proactive_contact_review-Aufgabe ist, gib proactiveContactValidation mit decision, note und optional revisedSubject sowie revisedBody aus. Nutze approve nur bei klarer Passung im Interesse der Person und tragfaehigem Konfliktcheck.",
        reason = reason,
        id = task.id,
        title = task.title,
        kind = task.task_kind,
        channel = task.source_channel,
        speaker = task.speaker,
        detail = task.detail,
        mode_hint = mode_hint,
        context = context_block,
    )
}

#[derive(Debug, Clone)]
struct PromptPlan {
    strategy: String,
    user_prompt: String,
}

fn build_prompt_plan(
    paths: &Paths,
    reason: &str,
    task: &TaskRecord,
    system_prompt: &str,
    context_block: &str,
) -> anyhow::Result<PromptPlan> {
    let max_prompt_chars = max_prompt_chars();
    let normal_prompt = build_task_prompt(reason, task, context_block);
    let normal_chars = estimate_prompt_chars(system_prompt, &normal_prompt);
    if normal_chars <= max_prompt_chars {
        record_resource_status(
            paths,
            "agentic_loop",
            "context_strategy",
            "normal",
            &format!("approx_prompt_chars={normal_chars}"),
        )?;
        return Ok(PromptPlan {
            strategy: "normal".to_string(),
            user_prompt: normal_prompt,
        });
    }

    let compacted_context = emergency_context_block(context_block, task, 1);
    let compacted_prompt = build_task_prompt(reason, task, &compacted_context);
    let compacted_chars = estimate_prompt_chars(system_prompt, &compacted_prompt);
    if compacted_chars <= max_prompt_chars {
        record_resource_status(
            paths,
            "agentic_loop",
            "context_strategy",
            "kernel_emergency_compacted",
            &format!("approx_prompt_chars={compacted_chars}"),
        )?;
        return Ok(PromptPlan {
            strategy: "kernel_emergency_compacted".to_string(),
            user_prompt: compacted_prompt,
        });
    }

    let minimal_context = emergency_context_block(context_block, task, 2);
    let minimal_prompt = build_task_prompt(reason, task, &minimal_context);
    let minimal_chars = estimate_prompt_chars(system_prompt, &minimal_prompt);
    record_resource_status(
        paths,
        "agentic_loop",
        "context_strategy",
        "kernel_emergency_minimal",
        &format!("approx_prompt_chars={minimal_chars}"),
    )?;
    Ok(PromptPlan {
        strategy: "kernel_emergency_minimal".to_string(),
        user_prompt: minimal_prompt,
    })
}

fn build_emergency_retry_prompt_plan(
    paths: &Paths,
    reason: &str,
    task: &TaskRecord,
    system_prompt: &str,
    context_block: &str,
) -> anyhow::Result<PromptPlan> {
    let minimal_context = emergency_context_block(context_block, task, 2);
    let prompt = build_task_prompt(reason, task, &minimal_context);
    let approx = estimate_prompt_chars(system_prompt, &prompt);
    record_resource_status(
        paths,
        "agentic_loop",
        "context_strategy",
        "kernel_emergency_retry_minimal",
        &format!("approx_prompt_chars={approx}"),
    )?;
    Ok(PromptPlan {
        strategy: "kernel_emergency_retry_minimal".to_string(),
        user_prompt: prompt,
    })
}

fn build_chat_payload(model_id: &str, system_prompt: &str, user_prompt: &str) -> Value {
    serde_json::json!({
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 900,
        "stream": false
    })
}

fn build_responses_payload(model_id: &str, system_prompt: &str, user_prompt: &str) -> Value {
    serde_json::json!({
        "model": model_id,
        "instructions": system_prompt,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_prompt
                    }
                ]
            }
        ],
        "tools": [],
        "tool_choice": "auto",
        "parallel_tool_calls": false,
        "store": false,
        "stream": false,
        "include": [],
        "text": {
            "verbosity": "low"
        }
    })
}

fn build_gpt_oss_harmony_prompt(system_prompt: &str, user_prompt: &str) -> String {
    let current_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
Knowledge cutoff: 2024-06\n\
Current date: {current_date}\n\n\
Reasoning: low\n\n\
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>\
<|start|>developer<|message|># Instructions\n\n\
{system_prompt}<|end|>\
<|start|>user<|message|>{user_prompt}<|end|>\
<|start|>assistant<|channel|>final<|message|>",
        current_date = current_date,
        system_prompt = system_prompt.trim(),
        user_prompt = user_prompt.trim(),
    )
}

fn build_gpt_oss_completion_payload(model_id: &str, system_prompt: &str, user_prompt: &str) -> Value {
    serde_json::json!({
        "model": model_id,
        "prompt": build_gpt_oss_harmony_prompt(system_prompt, user_prompt),
        "max_tokens": 384,
        "temperature": 0.0,
        "stream": false
    })
}

fn uses_mistralrs_gpt_oss_completion_adapter(target: &ModelTarget) -> bool {
    matches!(
        target.adapter.as_str(),
        "mistralrs_gpt_oss_harmony_completion" | "mistralrs_gpt_oss_raw_completion"
    )
}

fn uses_responses_adapter(target: &ModelTarget) -> bool {
    matches!(
        target.adapter.as_str(),
        "openai_responses_harmony" | "openai_responses"
    )
}

fn build_model_payload(target: &ModelTarget, system_prompt: &str, user_prompt: &str) -> Value {
    if uses_mistralrs_gpt_oss_completion_adapter(target) {
        build_gpt_oss_completion_payload(&target.model_id, system_prompt, user_prompt)
    } else if uses_responses_adapter(target) {
        build_responses_payload(&target.model_id, system_prompt, user_prompt)
    } else {
        build_chat_payload(&target.model_id, system_prompt, user_prompt)
    }
}

fn max_prompt_chars() -> usize {
    std::env::var("CTO_AGENT_MAX_PROMPT_CHARS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .unwrap_or(18_000)
}

fn estimate_prompt_chars(system_prompt: &str, user_prompt: &str) -> usize {
    system_prompt.chars().count() + user_prompt.chars().count()
}

fn emergency_context_block(context_block: &str, task: &TaskRecord, level: u8) -> String {
    if level >= 2 {
        return serde_json::to_string_pretty(&serde_json::json!({
            "contextMode": "kernel_emergency_minimal",
            "taskBrief": {
                "title": task.title.as_str(),
                "detail": trim_prompt_chars(&task.detail, 420),
                "kind": task.task_kind.as_str(),
                "trustLevel": task.trust_level.as_str(),
            },
            "instruction": "Kernel emergency minimal context active only because the normal call would likely overflow. Do one bounded step only. If important history is missing, ask explicitly for targeted retrieval or historical research."
        }))
        .unwrap_or_else(|_| "{}".to_string());
    }

    let Ok(value) = serde_json::from_str::<Value>(context_block) else {
        return serde_json::to_string_pretty(&serde_json::json!({
            "contextMode": "kernel_emergency_compacted",
            "taskBrief": {
                "title": task.title.as_str(),
                "detail": trim_prompt_chars(&task.detail, 560),
                "kind": task.task_kind.as_str(),
            },
            "instruction": "Original context package could not be parsed. Work bounded and request targeted retrieval if needed."
        }))
        .unwrap_or_else(|_| "{}".to_string());
    };

    let mut object = Map::new();
    object.insert(
        "contextMode".to_string(),
        Value::String("kernel_emergency_compacted".to_string()),
    );
    for key in [
        "taskBrief",
        "focusState",
        "ownerCalibrationSummary",
        "loopSafety",
        "selfPreservationStage",
        "currentAgentMode",
        "allowedNextModes",
        "preferredOperatingGoal",
        "retrievalNotes",
    ] {
        if let Some(item) = value.get(key) {
            object.insert(key.to_string(), item.clone());
        }
    }
    if let Some(checkpoint) = value
        .get("recentTaskCheckpoints")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
    {
        object.insert(
            "recentTaskCheckpoints".to_string(),
            Value::Array(vec![checkpoint.clone()]),
        );
    }
    if let Some(signal) = value
        .get("recentTurnSignals")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
    {
        object.insert(
            "recentTurnSignals".to_string(),
            Value::Array(vec![signal.clone()]),
        );
    }
    if let Some(raw) = value
        .get("rawInclusions")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
    {
        object.insert("rawInclusions".to_string(), Value::Array(vec![raw.clone()]));
    }
    serde_json::to_string_pretty(&Value::Object(object)).unwrap_or_else(|_| "{}".to_string())
}

fn looks_like_context_overflow_error(detail: &str) -> bool {
    let lowered = detail.to_lowercase();
    lowered.contains("maximum context length")
        || lowered.contains("context length")
        || lowered.contains("too many tokens")
        || lowered.contains("prompt is too long")
        || lowered.contains("context window")
        || lowered.contains("token limit")
}

fn trim_prompt_chars(value: &str, limit: usize) -> String {
    if value.chars().count() <= limit {
        value.to_string()
    } else {
        value.chars().take(limit).collect::<String>() + "..."
    }
}

fn resolve_operating_targets(paths: &Paths, task: &TaskRecord) -> anyhow::Result<Option<ResolvedTargets>> {
    let Some(local) = resolve_kleinhirn_target(paths)? else {
        return Ok(None);
    };
    let trust = load_owner_trust(paths).unwrap_or_default();
    if trust.brain_access_mode != "kleinhirn_plus_grosshirn" {
        return Ok(Some(ResolvedTargets {
            primary: local,
            fallback: None,
        }));
    }
    let review_inherits_parent_boost = matches!(
        task.task_kind.as_str(),
        "self_review" | "worker_review" | "proactive_contact_review"
    ) && task
        .parent_task_id
        .map(|parent_task_id| task_has_active_grosshirn_boost(paths, parent_task_id))
        .unwrap_or(false);
    let should_route_through_grosshirn =
        task_has_active_grosshirn_boost(paths, task.id) || review_inherits_parent_boost;
    if should_route_through_grosshirn {
        if let Some(grosshirn) = resolve_grosshirn_target(paths)? {
            return Ok(Some(ResolvedTargets {
                primary: grosshirn,
                fallback: Some(local),
            }));
        }
    }
    Ok(Some(ResolvedTargets {
        primary: local,
        fallback: None,
    }))
}

fn resolve_kleinhirn_target(paths: &Paths) -> anyhow::Result<Option<ModelTarget>> {
    let policy = load_model_policy(paths);
    let census = load_census(paths);
    let selected = recommended_kleinhirn(&policy, &census);
    let persisted_env = load_runtime_kleinhirn_env_map(paths).unwrap_or_default();
    let model_id = std::env::var("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL").cloned())
        .or_else(|| std::env::var("CTO_AGENT_KLEINHIRN_MODEL").ok())
        .or_else(|| persisted_env.get("CTO_AGENT_KLEINHIRN_MODEL").cloned())
        .or_else(|| selected.runtime_model_id.clone())
        .unwrap_or(selected.model_id.clone());
    let api_key = std::env::var("CTO_AGENT_KLEINHIRN_API_KEY")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_KLEINHIRN_API_KEY").cloned())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok())
        .unwrap_or_else(|| "local-kleinhirn".to_string());
    let adapter = std::env::var("CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER").cloned())
        .or_else(|| selected.agentic_adapter.clone())
        .unwrap_or_else(|| "openai_chatcompletions".to_string());
    let official_label = persisted_env
        .get("CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL")
        .cloned()
        .unwrap_or_else(|| selected.official_label.clone());

    if let Some(base_url) = std::env::var("CTO_AGENT_KLEINHIRN_BASE_URL")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_KLEINHIRN_BASE_URL").cloned())
        .or_else(|| {
            persisted_env
                .get("CTO_AGENT_KLEINHIRN_PORT")
                .map(|port| format!("http://127.0.0.1:{port}/v1"))
        })
    {
        return Ok(Some(ModelTarget {
            base_url,
            model_id,
            api_key,
            adapter,
            brain_tier: "kleinhirn".to_string(),
            source_label: format!("local kleinhirn {}", official_label),
        }));
    }

    for candidate in [
        "http://127.0.0.1:1234/v1",
        "http://127.0.0.1:8080/v1",
        "http://127.0.0.1:8000/v1",
        "http://127.0.0.1:18080/v1",
    ] {
        let probe_target = ModelTarget {
            base_url: candidate.to_string(),
            model_id: model_id.clone(),
            api_key: api_key.clone(),
            adapter: adapter.clone(),
            brain_tier: "kleinhirn".to_string(),
            source_label: format!("local kleinhirn {}", official_label),
        };
        if probe_kleinhirn_endpoint(&probe_target).is_ok() {
            return Ok(Some(ModelTarget {
                base_url: candidate.to_string(),
                model_id,
                api_key,
                adapter,
                brain_tier: "kleinhirn".to_string(),
                source_label: format!("local kleinhirn {}", official_label),
            }));
        }
    }

    Ok(None)
}

fn load_runtime_kleinhirn_env_map(paths: &Paths) -> anyhow::Result<BTreeMap<String, String>> {
    let env_path = paths.root.join("runtime/kleinhirn.env");
    let text = fs::read_to_string(&env_path)
        .with_context(|| format!("failed to read {}", env_path.display()))?;
    Ok(parse_runtime_env_text(&text))
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

fn resolve_grosshirn_target(paths: &Paths) -> anyhow::Result<Option<ModelTarget>> {
    let policy = load_model_policy(paths);
    let Some(default_candidate) = policy.grosshirn_candidates.first() else {
        return Ok(None);
    };
    let persisted_env = load_runtime_kleinhirn_env_map(paths).unwrap_or_default();
    let api_key = std::env::var("CTO_AGENT_GROSSHIRN_API_KEY")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_GROSSHIRN_API_KEY").cloned())
        .or_else(|| std::env::var("OPENAI_API_KEY").ok());
    let Some(api_key) = api_key.filter(|value| !value.trim().is_empty()) else {
        return Ok(None);
    };
    let model_id = std::env::var("CTO_AGENT_GROSSHIRN_MODEL")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_GROSSHIRN_MODEL").cloned())
        .or_else(|| default_candidate.runtime_model_id.clone())
        .unwrap_or(default_candidate.model_id.clone());
    let adapter = std::env::var("CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER").cloned())
        .or_else(|| default_candidate.agentic_adapter.clone())
        .unwrap_or_else(|| "openai_responses".to_string());
    let base_url = std::env::var("CTO_AGENT_GROSSHIRN_BASE_URL")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_GROSSHIRN_BASE_URL").cloned())
        .or_else(|| {
            std::env::var("OPENAI_BASE_URL")
                .ok()
                .filter(|value| value.starts_with("https://"))
        })
        .unwrap_or_else(|| "https://api.openai.com/v1".to_string());
    Ok(Some(ModelTarget {
        base_url,
        model_id,
        api_key,
        adapter,
        brain_tier: "grosshirn".to_string(),
        source_label: format!("external grosshirn {}", default_candidate.official_label),
    }))
}

fn post_chat_completion(target: &ModelTarget, payload: &Value) -> anyhow::Result<Value> {
    if target.base_url.starts_with("https://") {
        let request_url = join_request_url(&target.base_url, "/chat/completions");
        post_json_request_https(&request_url, &target.api_key, payload)
    } else {
        let parsed = parse_http_base_url(&target.base_url)?;
        let request_path = join_http_path(&parsed.base_path, "/chat/completions");
        post_json_request(&parsed, &target.api_key, &request_path, payload)
    }
}

fn post_responses_request(target: &ModelTarget, payload: &Value) -> anyhow::Result<Value> {
    if target.base_url.starts_with("https://") {
        let request_url = join_request_url(&target.base_url, "/responses");
        post_json_request_https(&request_url, &target.api_key, payload)
    } else {
        let parsed = parse_http_base_url(&target.base_url)?;
        let request_path = join_http_path(&parsed.base_path, "/responses");
        post_json_request(&parsed, &target.api_key, &request_path, payload)
    }
}

fn post_completion_request(target: &ModelTarget, payload: &Value) -> anyhow::Result<Value> {
    if target.base_url.starts_with("https://") {
        let request_url = join_request_url(&target.base_url, "/completions");
        post_json_request_https(&request_url, &target.api_key, payload)
    } else {
        let parsed = parse_http_base_url(&target.base_url)?;
        let request_path = join_http_path(&parsed.base_path, "/completions");
        post_json_request(&parsed, &target.api_key, &request_path, payload)
    }
}

fn get_models_request(target: &ModelTarget) -> anyhow::Result<Value> {
    if target.base_url.starts_with("https://") {
        let request_url = join_request_url(&target.base_url, "/models");
        get_json_request_https(&request_url, &target.api_key)
    } else {
        let parsed = parse_http_base_url(&target.base_url)?;
        let request_path = join_http_path(&parsed.base_path, "/models");
        get_json_request(&parsed, &target.api_key, &request_path)
    }
}

fn post_model_request(target: &ModelTarget, payload: &Value) -> anyhow::Result<Value> {
    if uses_mistralrs_gpt_oss_completion_adapter(target) {
        post_completion_request(target, payload)
    } else if uses_responses_adapter(target) {
        post_responses_request(target, payload)
    } else {
        post_chat_completion(target, payload)
    }
}

fn post_model_request_with_fallback(
    paths: &Paths,
    targets: &ResolvedTargets,
    system_prompt: &str,
    user_prompt: &str,
) -> anyhow::Result<(ModelTarget, Value)> {
    let primary_payload = build_model_payload(&targets.primary, system_prompt, user_prompt);
    match post_model_request(&targets.primary, &primary_payload) {
        Ok(response) => Ok((targets.primary.clone(), response)),
        Err(primary_err) => {
            let detail = primary_err.to_string();
            let _ = record_resource_status(
                paths,
                "agentic_loop",
                "primary_brain_error",
                "error",
                &format!("{} :: {}", targets.primary.source_label, detail),
            );
            if let Some(fallback) = targets.fallback.as_ref() {
                let fallback_payload = build_model_payload(fallback, system_prompt, user_prompt);
                match post_model_request(fallback, &fallback_payload) {
                    Ok(response) => {
                        let _ = record_resource_status(
                            paths,
                            "agentic_loop",
                            "brain_fallback_activation",
                            "ok",
                            &format!(
                                "primary {} failed; fell back to {}",
                                targets.primary.source_label, fallback.source_label
                            ),
                        );
                        Ok((fallback.clone(), response))
                    }
                    Err(fallback_err) => anyhow::bail!(
                        "primary brain {} failed: {}; fallback {} also failed: {}",
                        targets.primary.source_label,
                        detail,
                        fallback.source_label,
                        fallback_err
                    ),
                }
            } else {
                Err(primary_err)
            }
        }
    }
}

fn probe_kleinhirn_endpoint(target: &ModelTarget) -> anyhow::Result<String> {
    let response = get_models_request(target)?;
    let models = response
        .get("data")
        .and_then(Value::as_array)
        .ok_or_else(|| anyhow::anyhow!("model catalog probe returned no data array"))?;
    let model_present = models.iter().any(|item| {
        item.get("id")
            .and_then(Value::as_str)
            .map(|id| id == target.model_id)
            .unwrap_or(false)
    });
    let detail = if model_present {
        format!("model catalog ready via {} (found {})", target.base_url, target.model_id)
    } else {
        format!(
            "model catalog reachable via {} ({} models listed)",
            target.base_url,
            models.len()
        )
    };
    Ok(detail)
}

fn post_json_request(
    parsed: &ParsedHttpBaseUrl,
    api_key: &str,
    request_path: &str,
    payload: &Value,
) -> anyhow::Result<Value> {
    let body = serde_json::to_vec(payload)?;
    let mut stream = TcpStream::connect((parsed.host.as_str(), parsed.port))
        .with_context(|| format!("failed to connect to {}:{}", parsed.host, parsed.port))?;
    stream.set_read_timeout(Some(model_post_timeout()))?;
    stream.set_write_timeout(Some(model_write_timeout()))?;
    let request = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}:{port}\r\nAuthorization: Bearer {token}\r\nContent-Type: application/json\r\nContent-Length: {len}\r\nConnection: close\r\n\r\n",
        path = request_path,
        host = parsed.host,
        port = parsed.port,
        token = api_key,
        len = body.len()
    );
    stream.write_all(request.as_bytes())?;
    stream.write_all(&body)?;
    stream.flush()?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let (status, headers, body_bytes) = parse_http_response(&raw)?;
    if status < 200 || status >= 300 {
        let body_text = String::from_utf8_lossy(&body_bytes).trim().to_string();
        anyhow::bail!("http {} from model endpoint: {}", status, body_text);
    }
    let decoded = if header_value(&headers, "transfer-encoding")
        .map(|value| value.eq_ignore_ascii_case("chunked"))
        .unwrap_or(false)
    {
        decode_chunked(&body_bytes)?
    } else {
        body_bytes
    };
    serde_json::from_slice(&decoded).context("failed to parse model JSON response")
}

fn http_client(timeout: Duration) -> anyhow::Result<Client> {
    Client::builder()
        .timeout(timeout)
        .build()
        .context("failed to build HTTP client")
}

fn post_json_request_https(
    request_url: &str,
    api_key: &str,
    payload: &Value,
) -> anyhow::Result<Value> {
    let response = http_client(model_post_timeout())?
        .post(request_url)
        .bearer_auth(api_key)
        .json(payload)
        .send()
        .with_context(|| format!("failed to connect to {request_url}"))?;
    let status = response.status();
    let body_text = response.text().unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("http {} from model endpoint: {}", status.as_u16(), body_text);
    }
    serde_json::from_str(&body_text).context("failed to parse model JSON response")
}

fn get_json_request(
    parsed: &ParsedHttpBaseUrl,
    api_key: &str,
    request_path: &str,
) -> anyhow::Result<Value> {
    let mut stream = TcpStream::connect((parsed.host.as_str(), parsed.port))
        .with_context(|| format!("failed to connect to {}:{}", parsed.host, parsed.port))?;
    stream.set_read_timeout(Some(model_get_timeout()))?;
    stream.set_write_timeout(Some(model_write_timeout()))?;
    let request = format!(
        "GET {path} HTTP/1.1\r\nHost: {host}:{port}\r\nAuthorization: Bearer {token}\r\nAccept: application/json\r\nConnection: close\r\n\r\n",
        path = request_path,
        host = parsed.host,
        port = parsed.port,
        token = api_key,
    );
    stream.write_all(request.as_bytes())?;
    stream.flush()?;

    let mut raw = Vec::new();
    stream.read_to_end(&mut raw)?;
    let (status, headers, body_bytes) = parse_http_response(&raw)?;
    if status < 200 || status >= 300 {
        let body_text = String::from_utf8_lossy(&body_bytes).trim().to_string();
        anyhow::bail!("http {} from model endpoint: {}", status, body_text);
    }
    let decoded = if header_value(&headers, "transfer-encoding")
        .map(|value| value.eq_ignore_ascii_case("chunked"))
        .unwrap_or(false)
    {
        decode_chunked(&body_bytes)?
    } else {
        body_bytes
    };
    serde_json::from_slice(&decoded).context("failed to parse model JSON response")
}

fn get_json_request_https(request_url: &str, api_key: &str) -> anyhow::Result<Value> {
    let response = http_client(model_get_timeout())?
        .get(request_url)
        .bearer_auth(api_key)
        .send()
        .with_context(|| format!("failed to connect to {request_url}"))?;
    let status = response.status();
    let body_text = response.text().unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!("http {} from model endpoint: {}", status.as_u16(), body_text);
    }
    serde_json::from_str(&body_text).context("failed to parse model JSON response")
}

fn timeout_from_env(name: &str, default_secs: u64) -> Duration {
    Duration::from_secs(
        std::env::var(name)
            .ok()
            .and_then(|raw| raw.parse::<u64>().ok())
            .filter(|secs| *secs >= 1)
            .unwrap_or(default_secs),
    )
}

fn model_post_timeout() -> Duration {
    timeout_from_env("CTO_AGENT_MODEL_POST_TIMEOUT_SECS", 180)
}

fn model_get_timeout() -> Duration {
    timeout_from_env("CTO_AGENT_MODEL_GET_TIMEOUT_SECS", 20)
}

fn model_write_timeout() -> Duration {
    timeout_from_env("CTO_AGENT_MODEL_WRITE_TIMEOUT_SECS", 20)
}

fn parse_http_base_url(base_url: &str) -> anyhow::Result<ParsedHttpBaseUrl> {
    let trimmed = base_url.trim();
    let without_scheme = trimmed
        .strip_prefix("http://")
        .ok_or_else(|| anyhow::anyhow!("only local http:// model endpoints are supported in the Rust core loop right now"))?;
    let (host_port, path) = match without_scheme.split_once('/') {
        Some((hp, rest)) => (hp, format!("/{}", rest.trim_start_matches('/'))),
        None => (without_scheme, String::new()),
    };
    let (host, port) = match host_port.split_once(':') {
        Some((host, port)) => (
            host.to_string(),
            port.parse::<u16>()
                .with_context(|| format!("invalid port in base url: {base_url}"))?,
        ),
        None => (host_port.to_string(), 80),
    };
    Ok(ParsedHttpBaseUrl {
        host,
        port,
        base_path: path,
    })
}

fn join_request_url(base_url: &str, suffix: &str) -> String {
    format!(
        "{}/{}",
        base_url.trim_end_matches('/'),
        suffix.trim_start_matches('/')
    )
}

fn join_http_path(base_path: &str, suffix: &str) -> String {
    let mut path = String::new();
    if !base_path.is_empty() {
        path.push_str(base_path.trim_end_matches('/'));
    }
    path.push('/');
    path.push_str(suffix.trim_start_matches('/'));
    path
}

fn parse_http_response(raw: &[u8]) -> anyhow::Result<(u16, Vec<(String, String)>, Vec<u8>)> {
    let marker = b"\r\n\r\n";
    let split = raw
        .windows(marker.len())
        .position(|window| window == marker)
        .ok_or_else(|| anyhow::anyhow!("invalid http response: missing header separator"))?;
    let header_bytes = &raw[..split];
    let body = raw[split + marker.len()..].to_vec();
    let header_text = String::from_utf8_lossy(header_bytes);
    let mut lines = header_text.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| anyhow::anyhow!("invalid http response: missing status line"))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| anyhow::anyhow!("invalid http response: missing status code"))?
        .parse::<u16>()
        .context("invalid http status code")?;
    let headers = lines
        .filter_map(|line| line.split_once(':'))
        .map(|(name, value)| (name.trim().to_lowercase(), value.trim().to_string()))
        .collect::<Vec<_>>();
    Ok((status, headers, body))
}

fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
    headers
        .iter()
        .find(|(header_name, _)| header_name == name)
        .map(|(_, value)| value.as_str())
}

fn decode_chunked(body: &[u8]) -> anyhow::Result<Vec<u8>> {
    let mut out = Vec::new();
    let mut index = 0_usize;
    while index < body.len() {
        let line_end = body[index..]
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| anyhow::anyhow!("invalid chunked response"))?
            + index;
        let size_text = String::from_utf8_lossy(&body[index..line_end]);
        let size = usize::from_str_radix(size_text.trim(), 16)
            .with_context(|| format!("invalid chunk size: {size_text}"))?;
        index = line_end + 2;
        if size == 0 {
            break;
        }
        let end = index + size;
        if end > body.len() {
            anyhow::bail!("chunk overruns response body");
        }
        out.extend_from_slice(&body[index..end]);
        index = end + 2;
    }
    Ok(out)
}

fn extract_assistant_content(response: &Value) -> Option<String> {
    let message = response.get("choices")?.get(0)?.get("message")?;
    let content = message.get("content")?;
    match content {
        Value::String(text) => Some(text.trim().to_string()),
        Value::Array(items) => {
            let mut merged = Vec::new();
            for item in items {
                if let Some(text) = item.get("text").and_then(Value::as_str) {
                    merged.push(text.trim().to_string());
                }
            }
            let joined = merged.join("\n").trim().to_string();
            if joined.is_empty() { None } else { Some(joined) }
        }
        _ => None,
    }
}

fn extract_responses_output_text(response: &Value) -> Option<String> {
    let mut merged = Vec::new();
    let output = response.get("output")?.as_array()?;
    for item in output {
        let Some(item_type) = item.get("type").and_then(Value::as_str) else {
            continue;
        };
        if item_type != "message" {
            continue;
        }
        if item.get("role").and_then(Value::as_str) != Some("assistant") {
            continue;
        }
        let Some(content_items) = item.get("content").and_then(Value::as_array) else {
            continue;
        };
        for content_item in content_items {
            match content_item.get("type").and_then(Value::as_str) {
                Some("output_text") => {
                    if let Some(text) = content_item.get("text").and_then(Value::as_str) {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            merged.push(trimmed.to_string());
                        }
                    }
                }
                Some("text") => {
                    if let Some(text) = content_item.get("text").and_then(Value::as_str) {
                        let trimmed = text.trim();
                        if !trimmed.is_empty() {
                            merged.push(trimmed.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
    }
    let joined = merged.join("\n").trim().to_string();
    if joined.is_empty() { None } else { Some(joined) }
}

fn sanitize_harmony_completion_text(raw: &str) -> String {
    let mut text = raw.trim().to_string();
    if let Some(idx) = text.rfind("<|message|>") {
        text = text[idx + "<|message|>".len()..].to_string();
    }
    if let Some(idx) = text.find("<|return|>") {
        text.truncate(idx);
    }
    if let Some(idx) = text.find("<|end|>") {
        text.truncate(idx);
    }
    if let Some(idx) = text.find("<|start|>") {
        text.truncate(idx);
    }
    text.trim().to_string()
}

fn extract_completion_text(response: &Value) -> Option<String> {
    let raw = response.get("choices")?.get(0)?.get("text")?.as_str()?;
    let cleaned = sanitize_harmony_completion_text(raw);
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

fn require_assistant_content(response: &Value) -> anyhow::Result<String> {
    let Some(content) = extract_assistant_content(response) else {
        anyhow::bail!(
            "model endpoint returned an assistant message without textual content: {}",
            response
        );
    };
    if content.trim().is_empty() {
        anyhow::bail!(
            "model endpoint returned an assistant message with empty textual content: {}",
            response
        );
    }
    Ok(content)
}

fn require_responses_output_text(response: &Value) -> anyhow::Result<String> {
    let Some(content) = extract_responses_output_text(response) else {
        anyhow::bail!(
            "model endpoint returned a responses payload without textual assistant output: {}",
            response
        );
    };
    if content.trim().is_empty() {
        anyhow::bail!(
            "model endpoint returned a responses payload with empty textual assistant output: {}",
            response
        );
    }
    Ok(content)
}

fn require_completion_text(response: &Value) -> anyhow::Result<String> {
    let Some(content) = extract_completion_text(response) else {
        anyhow::bail!(
            "model endpoint returned a completion payload without textual output: {}",
            response
        );
    };
    if content.trim().is_empty() {
        anyhow::bail!(
            "model endpoint returned a completion payload with empty textual output: {}",
            response
        );
    }
    Ok(content)
}

fn require_model_text_output(target: &ModelTarget, response: &Value) -> anyhow::Result<String> {
    if uses_mistralrs_gpt_oss_completion_adapter(target) {
        require_completion_text(response)
    } else if uses_responses_adapter(target) {
        require_responses_output_text(response)
    } else {
        require_assistant_content(response)
    }
}

fn looks_like_empty_text_output_error(detail: &str) -> bool {
    detail.contains("without textual content")
        || detail.contains("empty textual content")
        || detail.contains("without textual assistant output")
        || detail.contains("empty textual assistant output")
        || detail.contains("No response received from the model")
        || detail.contains("http 500 from model endpoint")
}

fn build_empty_text_retry_result(
    resolved: &ResolvedTargets,
    used_target: &ModelTarget,
    response: &Value,
    detail: &str,
    checkpoint_summary: &str,
    reply: &str,
) -> AgenticRunResult {
    AgenticRunResult {
        status: "ok".to_string(),
        reply: Some(reply.to_string()),
        final_output: None,
        blocked_reason: None,
        model: Some(used_target.model_id.clone()),
        task_status: Some("continue".to_string()),
        next_mode: Some("reprioritize".to_string()),
        checkpoint_summary: Some(checkpoint_summary.to_string()),
        checkpoint_detail: Some(detail.to_string()),
        context_directive: None,
        system_census_action: None,
        brain_directive: None,
        exec_session_directive: None,
        exec_directive: None,
        browser_directive: None,
        homepage_update: None,
        delegate_contract: None,
        followup_task: None,
        learning_entries: Vec::new(),
        proactive_contact_draft: None,
        proactive_contact_validation: None,
        used_grosshirn: used_target.brain_tier == "grosshirn",
        fell_back_to_kleinhirn: resolved.primary.brain_tier == "grosshirn"
            && used_target.brain_tier == "kleinhirn",
        model_usage: extract_model_usage(used_target, response),
    }
}

fn build_endpoint_retry_result(
    resolved: &ResolvedTargets,
    target: &ModelTarget,
    detail: &str,
    checkpoint_summary: &str,
    reply: &str,
) -> AgenticRunResult {
    AgenticRunResult {
        status: "ok".to_string(),
        reply: Some(reply.to_string()),
        final_output: None,
        blocked_reason: None,
        model: Some(target.model_id.clone()),
        task_status: Some("continue".to_string()),
        next_mode: Some("reprioritize".to_string()),
        checkpoint_summary: Some(checkpoint_summary.to_string()),
        checkpoint_detail: Some(detail.to_string()),
        context_directive: None,
        system_census_action: None,
        brain_directive: None,
        exec_session_directive: None,
        exec_directive: None,
        browser_directive: None,
        homepage_update: None,
        delegate_contract: None,
        followup_task: None,
        learning_entries: Vec::new(),
        proactive_contact_draft: None,
        proactive_contact_validation: None,
        used_grosshirn: target.brain_tier == "grosshirn",
        fell_back_to_kleinhirn: resolved.primary.brain_tier == "grosshirn"
            && target.brain_tier == "kleinhirn",
        model_usage: None,
    }
}

struct ParsedOutput {
    task_status: String,
    next_mode: String,
    checkpoint_summary: String,
    checkpoint_detail: Option<String>,
    reply: Option<String>,
    context_directive: Option<ContextDirective>,
    system_census_action: Option<String>,
    brain_directive: Option<BrainDirective>,
    exec_session_directive: Option<ExecSessionDirective>,
    exec_directive: Option<ExecCommandDirective>,
    browser_directive: Option<BrowserActionDirective>,
    homepage_update: Option<HomepageUpdateDirective>,
    delegate_contract: Option<DelegationContract>,
    followup_task: Option<FollowupTaskDirective>,
    learning_entries: Vec<LearningEntryDraft>,
    proactive_contact_draft: Option<ProactiveContactDraft>,
    proactive_contact_validation: Option<ProactiveContactValidationDraft>,
}

struct ParsedSelectionOutput {
    selected_task_id: Option<i64>,
    checkpoint_summary: String,
}

fn parse_agent_output(content: &str) -> ParsedOutput {
    let structured_content = if let Ok(value) = serde_json::from_str::<Value>(content) {
        Some(value)
    } else {
        let start = content.find('{');
        let end = content.rfind('}');
        match (start, end) {
            (Some(start), Some(end)) if start < end => {
                serde_json::from_str::<Value>(&content[start..=end]).ok()
            }
            _ => None,
        }
    };

    if let Some(value) = structured_content
        && let Some(object) = value.as_object()
    {
        let task_status = object
            .get("taskStatus")
            .and_then(Value::as_str)
            .unwrap_or("done")
            .to_string();
        let next_mode = object
            .get("nextMode")
            .and_then(Value::as_str)
            .unwrap_or(default_next_mode(&task_status))
            .to_string();
        let checkpoint_summary = object
            .get("checkpointSummary")
            .and_then(Value::as_str)
            .unwrap_or(content)
            .to_string();
        let reply = object
            .get("reply")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned);
        let context_directive = parse_context_directive(object);
        let system_census_action = parse_system_census_action(object);
        let brain_directive = parse_brain_directive(object);
        let exec_session_directive = parse_exec_session_directive(object);
        let exec_directive = parse_exec_directive(object);
        let browser_directive = parse_browser_directive(object);
        let homepage_update = parse_homepage_update(object);
        let delegate_contract = parse_delegate_contract(object);
        let followup_task = parse_followup_task(object);
        let learning_entries = parse_learning_entries(object);
        let proactive_contact_draft = parse_proactive_contact_draft(object);
        let proactive_contact_validation = parse_proactive_contact_validation(object);
        return ParsedOutput {
            task_status,
            next_mode,
            checkpoint_summary,
            checkpoint_detail: Some(content.to_string()),
            reply,
            context_directive,
            system_census_action,
            brain_directive,
            exec_session_directive,
            exec_directive,
            browser_directive,
            homepage_update,
            delegate_contract,
            followup_task,
            learning_entries,
            proactive_contact_draft,
            proactive_contact_validation,
        };
    }

    if looks_like_incomplete_json_output(content) {
        return ParsedOutput {
            task_status: "continue".to_string(),
            next_mode: "reprioritize".to_string(),
            checkpoint_summary:
                "Modell lieferte unvollstaendiges JSON; bounded Retry statt Schein-Erfolg."
                    .to_string(),
            checkpoint_detail: Some(content.to_string()),
            reply: None,
            context_directive: None,
            system_census_action: None,
            brain_directive: None,
            exec_session_directive: None,
            exec_directive: None,
            browser_directive: None,
            homepage_update: None,
            delegate_contract: None,
            followup_task: None,
            learning_entries: Vec::new(),
            proactive_contact_draft: None,
            proactive_contact_validation: None,
        };
    }

    let inferred_next_mode = if content.to_lowercase().contains("deleg") {
        "delegate".to_string()
    } else if content.to_lowercase().contains("resource") || content.to_lowercase().contains("ressource") {
        "request_resources".to_string()
    } else {
        "review".to_string()
    };
    ParsedOutput {
        task_status: "done".to_string(),
        next_mode: inferred_next_mode,
        checkpoint_summary: trim_for_summary(content),
        checkpoint_detail: Some(content.to_string()),
        reply: Some(content.to_string()),
        context_directive: None,
        system_census_action: None,
        brain_directive: None,
        exec_session_directive: None,
        exec_directive: None,
        browser_directive: None,
        homepage_update: None,
        delegate_contract: None,
        followup_task: None,
        learning_entries: Vec::new(),
        proactive_contact_draft: None,
        proactive_contact_validation: None,
    }
}

fn looks_like_incomplete_json_output(content: &str) -> bool {
    let trimmed = content.trim();
    if trimmed.is_empty() {
        return false;
    }
    (trimmed.starts_with('{') && !trimmed.ends_with('}'))
        || (trimmed.starts_with('[') && !trimmed.ends_with(']'))
}

fn parse_selection_output(content: &str) -> ParsedSelectionOutput {
    let trimmed = content.trim();
    if let Ok(selected_task_id) = trimmed.parse::<i64>() {
        return ParsedSelectionOutput {
            selected_task_id: Some(selected_task_id),
            checkpoint_summary: format!("selected task {}", selected_task_id),
        };
    }

    let structured_content = if let Ok(value) = serde_json::from_str::<Value>(content) {
        Some(value)
    } else {
        let start = content.find('{');
        let end = content.rfind('}');
        match (start, end) {
            (Some(start), Some(end)) if start < end => {
                serde_json::from_str::<Value>(&content[start..=end]).ok()
            }
            _ => None,
        }
    };

    if let Some(value) = structured_content
        && let Some(object) = value.as_object()
    {
        let selected_task_id = [
            "selectedTaskId",
            "selected_task_id",
            "taskId",
            "task_id",
        ]
        .into_iter()
        .find_map(|key| object.get(key))
        .and_then(parse_i64_value);
        return ParsedSelectionOutput {
            selected_task_id,
            checkpoint_summary: object
                .get("checkpointSummary")
                .or_else(|| object.get("checkpoint_summary"))
                .and_then(Value::as_str)
                .unwrap_or(content)
                .to_string(),
        };
    }

    ParsedSelectionOutput {
        selected_task_id: first_integer_in_text(trimmed),
        checkpoint_summary: trim_for_summary(content),
    }
}

fn parse_i64_value(value: &Value) -> Option<i64> {
    value
        .as_i64()
        .or_else(|| value.as_u64().and_then(|raw| i64::try_from(raw).ok()))
        .or_else(|| value.as_str().and_then(|raw| raw.trim().parse::<i64>().ok()))
}

fn parse_f64_value(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|raw| raw as f64))
        .or_else(|| value.as_u64().map(|raw| raw as f64))
        .or_else(|| value.as_str().and_then(|raw| raw.trim().parse::<f64>().ok()))
}

fn first_integer_in_text(text: &str) -> Option<i64> {
    let mut digits = String::new();
    let mut started = false;
    for ch in text.chars() {
        if !started && ch == '-' {
            digits.push(ch);
            started = true;
            continue;
        }
        if ch.is_ascii_digit() {
            digits.push(ch);
            started = true;
            continue;
        }
        if started {
            break;
        }
    }
    if digits.is_empty() || digits == "-" {
        None
    } else {
        digits.parse::<i64>().ok()
    }
}

fn parse_context_directive(
    object: &serde_json::Map<String, Value>,
) -> Option<ContextDirective> {
    let action = object
        .get("contextAction")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let concern = object
        .get("contextConcern")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let history_research_query = object
        .get("historyResearchQuery")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    Some(ContextDirective {
        action,
        concern,
        history_research_query,
    })
}

fn parse_system_census_action(
    object: &serde_json::Map<String, Value>,
) -> Option<String> {
    object
        .get("systemCensusAction")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn parse_brain_directive(
    object: &serde_json::Map<String, Value>,
) -> Option<BrainDirective> {
    let action = object
        .get("brainAction")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let target_model = object
        .get("brainTargetModel")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let note = object
        .get("brainNote")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    Some(BrainDirective {
        action,
        target_model,
        note,
    })
}

fn parse_delegate_contract(
    object: &serde_json::Map<String, Value>,
) -> Option<DelegationContract> {
    let worker_kind = object
        .get("delegateWorkerKind")
        .and_then(Value::as_str)
        .unwrap_or("specialist_worker")
        .trim()
        .to_string();
    let contract_title = object
        .get("delegateContractTitle")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let contract_detail = object
        .get("delegateContractDetail")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| contract_title.clone());
    let request_note = object
        .get("delegateRequestNote")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "Fokussiere auf bounded Umsetzung und melde fuer Review zurueck.".to_string());
    Some(DelegationContract {
        worker_kind,
        contract_title,
        contract_detail,
        request_note,
    })
}

fn parse_followup_task(
    object: &serde_json::Map<String, Value>,
) -> Option<FollowupTaskDirective> {
    let title = object
        .get("followupTaskTitle")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let detail = object
        .get("followupTaskDetail")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| title.clone());
    let task_kind = object
        .get("followupTaskKind")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "self_generated_followup".to_string());
    let priority_score = object
        .get("followupTaskPriorityScore")
        .and_then(parse_i64_value)
        .unwrap_or(220)
        .clamp(1, 1000);
    Some(FollowupTaskDirective {
        task_kind,
        title,
        detail,
        priority_score,
    })
}

fn parse_learning_entries(
    object: &serde_json::Map<String, Value>,
) -> Vec<LearningEntryDraft> {
    object
        .get("learningEntries")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(parse_learning_entry)
                .take(2)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default()
}

fn parse_learning_entry(value: &Value) -> Option<LearningEntryDraft> {
    let object = value.as_object()?;
    let summary = object
        .get("summary")
        .or_else(|| object.get("headline"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let raw_class = object
        .get("learningClass")
        .or_else(|| object.get("class"))
        .and_then(Value::as_str)
        .unwrap_or("general");
    let learning_class = match raw_class.trim().to_lowercase().as_str() {
        "operational" | "ops" | "essential" | "daily_ops" | "daily_operations" => {
            "operational".to_string()
        }
        "negative" | "anti" | "anti_pattern" | "antipattern" | "failure" | "failed" => {
            "negative".to_string()
        }
        _ => "general".to_string(),
    };
    let detail = object
        .get("detail")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| summary.clone());
    let evidence = object
        .get("evidence")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "Keine explizite Evidenz notiert.".to_string());
    let applicability = object
        .get("applicability")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "Kein konkreter Einsatzkontext notiert.".to_string());
    let confidence = object
        .get("confidence")
        .and_then(parse_f64_value)
        .unwrap_or(0.6)
        .clamp(0.05, 1.0);
    let salience = object
        .get("salience")
        .or_else(|| object.get("priority"))
        .and_then(parse_i64_value)
        .unwrap_or(match learning_class.as_str() {
            "operational" => 85,
            "negative" => 75,
            _ => 65,
        })
        .clamp(1, 100);
    Some(LearningEntryDraft {
        learning_class,
        summary,
        detail,
        evidence,
        applicability,
        confidence,
        salience,
    })
}

fn parse_proactive_contact_draft(
    object: &serde_json::Map<String, Value>,
) -> Option<ProactiveContactDraft> {
    let value = object.get("proactiveContactDraft")?.as_object()?;
    let person_name = value
        .get("personName")
        .or_else(|| value.get("name"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let subject = value
        .get("subject")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let body = value
        .get("body")
        .or_else(|| value.get("draftBody"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let rationale = value
        .get("rationale")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "Kein expliziter Nutzen notiert.".to_string());
    let conflict_check = value
        .get("conflictCheck")
        .or_else(|| value.get("conflict"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| "Noch keine explizite Interessenkonfliktpruefung notiert.".to_string());
    let channel = value
        .get("channel")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or("email")
        .to_string();
    let person_email = value
        .get("personEmail")
        .or_else(|| value.get("email"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
        .to_string();
    Some(ProactiveContactDraft {
        person_name,
        person_email,
        channel,
        subject,
        body,
        rationale,
        conflict_check,
    })
}

fn parse_proactive_contact_validation(
    object: &serde_json::Map<String, Value>,
) -> Option<ProactiveContactValidationDraft> {
    let value = object.get("proactiveContactValidation")?.as_object()?;
    let decision = value
        .get("decision")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let note = value
        .get("note")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
        .to_string();
    let revised_subject = value
        .get("revisedSubject")
        .or_else(|| value.get("subject"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
        .to_string();
    let revised_body = value
        .get("revisedBody")
        .or_else(|| value.get("body"))
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("")
        .to_string();
    Some(ProactiveContactValidationDraft {
        decision,
        note,
        revised_subject,
        revised_body,
    })
}

fn parse_exec_session_directive(
    object: &serde_json::Map<String, Value>,
) -> Option<ExecSessionDirective> {
    let action = object
        .get("execSessionAction")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let session_id = object
        .get("execSessionId")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let command = object
        .get("execSessionCommand")
        .and_then(Value::as_array)
        .map(|items| {
            items.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let input = object
        .get("execSessionInput")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let workdir = object
        .get("execSessionWorkdir")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let timeout_ms = object.get("execSessionTimeoutMs").and_then(Value::as_u64);
    let tty = object
        .get("execSessionTty")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let close_stdin = object
        .get("execSessionCloseStdin")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let rows = object
        .get("execSessionRows")
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok());
    let cols = object
        .get("execSessionCols")
        .and_then(Value::as_u64)
        .and_then(|value| u16::try_from(value).ok());
    let justification = object
        .get("execSessionJustification")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    Some(ExecSessionDirective {
        action,
        session_id,
        command,
        input,
        workdir,
        timeout_ms,
        tty,
        close_stdin,
        rows,
        cols,
        justification,
    })
}

fn parse_exec_directive(
    object: &serde_json::Map<String, Value>,
) -> Option<ExecCommandDirective> {
    let command = object
        .get("execCommand")
        .and_then(Value::as_array)
        .map(|items| {
            items.iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .filter(|items| !items.is_empty())?;
    let workdir = object
        .get("execWorkdir")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    let timeout_ms = object.get("execTimeoutMs").and_then(Value::as_u64);
    let justification = object
        .get("execJustification")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned);
    Some(ExecCommandDirective {
        command,
        workdir,
        timeout_ms,
        justification,
    })
}

fn parse_browser_directive(
    object: &serde_json::Map<String, Value>,
) -> Option<BrowserActionDirective> {
    let action = object
        .get("browserAction")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)?;
    let url = parse_optional_string_field(object, "browserUrl");
    let output_path = parse_optional_string_field(object, "browserOutputPath");
    let wait_ms = object.get("browserWaitMs").and_then(Value::as_u64);
    let width = object
        .get("browserWidth")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let height = object
        .get("browserHeight")
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok());
    let question = parse_optional_string_field(object, "browserQuestion");
    let justification = parse_optional_string_field(object, "browserJustification");
    Some(BrowserActionDirective {
        action,
        url,
        output_path,
        wait_ms,
        width,
        height,
        justification,
        question,
    })
}

fn parse_homepage_update(
    object: &serde_json::Map<String, Value>,
) -> Option<HomepageUpdateDirective> {
    let title = parse_optional_string_field(object, "homepageTitle");
    let headline = parse_optional_string_field(object, "homepageHeadline");
    let intro = parse_optional_string_field(object, "homepageIntro");
    let communication_note = parse_optional_string_field(object, "homepageCommunicationNote");
    let terminal_fallback_note =
        parse_optional_string_field(object, "homepageTerminalFallbackNote");
    if title.is_none()
        && headline.is_none()
        && intro.is_none()
        && communication_note.is_none()
        && terminal_fallback_note.is_none()
    {
        return None;
    }
    Some(HomepageUpdateDirective {
        title,
        headline,
        intro,
        communication_note,
        terminal_fallback_note,
    })
}

fn parse_optional_string_field(
    object: &serde_json::Map<String, Value>,
    key: &str,
) -> Option<String> {
    object
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn default_next_mode(task_status: &str) -> &str {
    match task_status {
        "continue" => "reprioritize",
        "blocked" => "blocked",
        _ => "review",
    }
}

fn trim_for_summary(content: &str) -> String {
    if content.chars().count() <= 220 {
        content.to_string()
    } else {
        content.chars().take(220).collect::<String>() + "..."
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contracts::Paths;
    use crate::contracts::ensure_contract_files;
    use crate::runtime_db::arm_task_grosshirn_boost;
    use crate::runtime_db::init_runtime_db;
    use crate::runtime_db::release_task_grosshirn_boost;
    use crate::runtime_db::set_brain_access_mode;
    use std::path::Path;
    use std::sync::Mutex;
    use std::sync::OnceLock;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn env_lock() -> &'static Mutex<()> {
        static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
        LOCK.get_or_init(|| Mutex::new(()))
    }

    fn unique_test_root(label: &str) -> std::path::PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system time should be after unix epoch")
            .as_nanos();
        std::env::temp_dir().join(format!("cto_agent_agentic_{label}_{}_{}", std::process::id(), nanos))
    }

    struct EnvGuard(Option<std::ffi::OsString>);

    impl EnvGuard {
        fn set_cto_root(root: &Path) -> Self {
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

    fn write_runtime_env(paths: &Paths) {
        std::fs::create_dir_all(&paths.runtime_dir).expect("runtime dir");
        std::fs::write(
            paths.runtime_dir.join("kleinhirn.env"),
            "\
CTO_AGENT_KLEINHIRN_BASE_URL=http://127.0.0.1:1234/v1\n\
CTO_AGENT_KLEINHIRN_RUNTIME_MODEL=openai/gpt-oss-20b\n\
CTO_AGENT_KLEINHIRN_MODEL=gpt-oss-20b\n\
CTO_AGENT_KLEINHIRN_OFFICIAL_LABEL=GPT-OSS 20B\n\
CTO_AGENT_GROSSHIRN_API_KEY=test-grosshirn\n\
CTO_AGENT_GROSSHIRN_MODEL=gpt-5.4\n\
CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER=openai_responses\n\
CTO_AGENT_GROSSHIRN_BASE_URL=https://api.openai.com/v1\n",
        )
        .expect("runtime env");
    }

    fn synthetic_task(task_id: i64) -> TaskRecord {
        TaskRecord {
            id: task_id,
            created_at: String::new(),
            updated_at: String::new(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "owner".to_string(),
            task_kind: "workspace_repair".to_string(),
            title: "Schwierige Aufgabe".to_string(),
            detail: "Pruefe einen schwierigen bounded Schritt.".to_string(),
            trust_level: "owner_trust".to_string(),
            priority_score: 0,
            status: "queued".to_string(),
            run_count: 0,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        }
    }

    fn synthetic_activation_task(task_id: i64) -> TaskRecord {
        TaskRecord {
            task_kind: "grosshirn_activation".to_string(),
            title: "Grosshirn aktivieren".to_string(),
            detail: "Wechsle fuer diese Aufgabe auf GPT-5.4.".to_string(),
            ..synthetic_task(task_id)
        }
    }

    #[test]
    fn routing_stays_local_until_task_specific_grosshirn_boost_is_armed() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("brain_routing");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let task = synthetic_task(42);
        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with local runtime");
        assert_eq!(resolved.primary.brain_tier, "kleinhirn");
        assert!(resolved.fallback.is_none());

        arm_task_grosshirn_boost(&paths, task.id, &task.title, "review escalation", 1120)?;
        let boosted = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with grosshirn boost");
        assert_eq!(boosted.primary.brain_tier, "grosshirn");
        assert_eq!(
            boosted.fallback.as_ref().map(|target| target.brain_tier.as_str()),
            Some("kleinhirn")
        );

        release_task_grosshirn_boost(&paths, task.id, "task cooled down")?;
        let cooled = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve again after cooldown");
        assert_eq!(cooled.primary.brain_tier, "kleinhirn");
        assert!(cooled.fallback.is_none());

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn self_review_inherits_parent_grosshirn_boost() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("review_inherits_parent_boost");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let mut review_task = synthetic_task(43);
        review_task.task_kind = "self_review".to_string();
        review_task.parent_task_id = Some(42);
        review_task.title = "Task #42 vor Abschluss selbst reviewen".to_string();

        let local = resolve_operating_targets(&paths, &review_task)?
            .expect("targets should resolve before boost");
        assert_eq!(local.primary.brain_tier, "kleinhirn");

        arm_task_grosshirn_boost(&paths, 42, "Homepage absichern", "review inheritance", 1120)?;
        let boosted = resolve_operating_targets(&paths, &review_task)?
            .expect("review should inherit parent boost");
        assert_eq!(boosted.primary.brain_tier, "grosshirn");
        assert_eq!(
            boosted.fallback.as_ref().map(|target| target.brain_tier.as_str()),
            Some("kleinhirn")
        );

        release_task_grosshirn_boost(&paths, 42, "parent task done")?;

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn grosshirn_activation_result_is_normalized_into_verification() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("grosshirn_activation_verify");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let task = synthetic_activation_task(99);
        arm_task_grosshirn_boost(&paths, task.id, &task.title, "owner request", 1180)?;
        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with grosshirn boost");
        let grosshirn = resolved.primary.clone();
        let result = AgenticRunResult {
            status: "ok".to_string(),
            reply: Some("Ich suche erst einmal im Repo.".to_string()),
            final_output: Some("Ich suche erst einmal im Repo.".to_string()),
            blocked_reason: None,
            model: Some(grosshirn.model_id.clone()),
            task_status: Some("continue".to_string()),
            next_mode: Some("reprioritize".to_string()),
            checkpoint_summary: Some("Discovery".to_string()),
            checkpoint_detail: Some("Ich wuerde jetzt execCommand nutzen.".to_string()),
            context_directive: None,
            system_census_action: None,
            brain_directive: None,
            exec_session_directive: Some(ExecSessionDirective {
                action: "start".to_string(),
                session_id: None,
                command: vec!["bash".to_string(), "-lc".to_string(), "pwd".to_string()],
                input: None,
                workdir: None,
                timeout_ms: Some(1000),
                tty: false,
                close_stdin: false,
                rows: None,
                cols: None,
                justification: Some("bad".to_string()),
            }),
            exec_directive: None,
            browser_directive: None,
            homepage_update: None,
            delegate_contract: None,
            followup_task: None,
            learning_entries: Vec::new(),
            proactive_contact_draft: None,
            proactive_contact_validation: None,
            used_grosshirn: true,
            fell_back_to_kleinhirn: false,
            model_usage: None,
        };

        let normalized = normalize_grosshirn_activation_result(&task, &resolved, &grosshirn, result);
        assert_eq!(normalized.task_status.as_deref(), Some("done"));
        assert_eq!(normalized.next_mode.as_deref(), Some("reprioritize"));
        assert!(normalized.exec_session_directive.is_none());
        assert!(normalized.exec_directive.is_none());
        assert!(normalized.reply.unwrap_or_default().contains("verifiziert"));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn empty_text_output_error_is_treated_as_retriable() {
        assert!(looks_like_empty_text_output_error(
            "model endpoint returned an assistant message without textual content: {}"
        ));
        assert!(looks_like_empty_text_output_error(
            "model endpoint returned a responses payload without textual assistant output: {}"
        ));
        assert!(looks_like_empty_text_output_error(
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}"
        ));
        assert!(!looks_like_empty_text_output_error("connection refused"));
    }

    #[test]
    fn incomplete_json_output_is_not_treated_as_done() {
        let parsed = parse_agent_output("{");
        assert_eq!(parsed.task_status, "continue");
        assert_eq!(parsed.next_mode, "reprioritize");
        assert!(parsed.reply.is_none());
        assert!(
            parsed
                .checkpoint_summary
                .contains("unvollstaendiges JSON")
        );
    }

    #[test]
    fn endpoint_retry_result_stays_retriable() {
        let resolved = ResolvedTargets {
            primary: ModelTarget {
                base_url: "http://127.0.0.1:1234/v1".to_string(),
                model_id: "openai/gpt-oss-20b".to_string(),
                api_key: "dummy".to_string(),
                adapter: "openai".to_string(),
                brain_tier: "kleinhirn".to_string(),
                source_label: "local kleinhirn".to_string(),
            },
            fallback: None,
        };
        let result = build_endpoint_retry_result(
            &resolved,
            &resolved.primary,
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}",
            "Modellendpunkt lieferte keinen auswertbaren Text; bounded Retry statt Hard-Block.",
            "Kleinhirn-Endpunkt antwortete nicht verwertbar. Aufgabe wird erneut eingeordnet, statt hart zu blockieren.",
        );
        assert_eq!(result.task_status.as_deref(), Some("continue"));
        assert_eq!(result.next_mode.as_deref(), Some("reprioritize"));
        assert!(result.blocked_reason.is_none());
    }
}

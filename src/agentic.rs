use crate::browser_engine::BrowserActionDirective;
use crate::context_controller::ContextPreparedArtifact;
use crate::contracts::Paths;
use crate::contracts::load_bios;
use crate::contracts::load_census;
use crate::contracts::load_model_policy;
use crate::contracts::normalize_runtime_model_choice;
use crate::contracts::recommended_kleinhirn;
use crate::runtime_db::LearningEntryDraft;
use crate::runtime_db::ProactiveContactDraft;
use crate::runtime_db::ProactiveContactValidationDraft;
use crate::runtime_db::TaskRecord;
use crate::runtime_db::activate_selected_task;
use crate::runtime_db::has_open_loop_incident;
use crate::runtime_db::latest_context_package_for_task;
use crate::runtime_db::list_recent_context_packages_for_task;
use crate::runtime_db::list_queued_tasks;
use crate::runtime_db::list_task_checkpoints;
use crate::runtime_db::load_active_task;
use crate::runtime_db::load_owner_trust;
use crate::runtime_db::record_resource_status;
use crate::runtime_db::requeue_retryable_blocked_tasks_after_runtime_stall;
use crate::runtime_db::select_next_task;
use crate::runtime_db::task_has_active_grosshirn_boost;
use crate::runtime_db::yield_active_task_for_preemption;
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
    pub completion_review: Option<CompletionReviewDirective>,
    pub prepared_context_artifact: Option<ContextPreparedArtifact>,
    pub used_grosshirn: bool,
    pub fell_back_to_kleinhirn: bool,
    pub retriable_local_failure: bool,
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
            completion_review: None,
            prepared_context_artifact: None,
            used_grosshirn: false,
            fell_back_to_kleinhirn: false,
            retriable_local_failure: false,
            model_usage: None,
        }
    }

    pub fn best_reply(&self) -> Option<&str> {
        self.reply.as_deref().or(self.final_output.as_deref())
    }

    pub fn status_note(&self) -> String {
        match (
            &self.status[..],
            self.blocked_reason.as_deref(),
            self.model.as_deref(),
        ) {
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
    pub duration_ms: Option<i64>,
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

#[derive(Debug, Clone, Deserialize)]
pub struct CompletionReviewDirective {
    pub decision: String,
    pub note: String,
    pub evidence_gaps: Vec<String>,
    pub confidence: Option<f64>,
}

#[derive(Debug, Clone)]
struct ModelTarget {
    base_url: String,
    model_id: String,
    api_key: String,
    adapter: String,
    brain_tier: String,
    source_label: String,
    reasoning_effort: String,
}

#[derive(Debug, Clone)]
struct ResolvedTargets {
    primary: ModelTarget,
    fallback: Option<ModelTarget>,
}

#[derive(Debug, Clone)]
struct CompactRoutingPreference {
    tier: String,
    requested_model: String,
    switch_planned: bool,
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
            "missing_kleinhirn_endpoint: set CTO_AGENT_KLEINHIRN_BASE_URL or run a local OpenAI-compatible kleinhirn endpoint",
        ));
    };
    let target = resolved.primary.clone();

    record_resource_status(
        paths,
        "agentic_loop",
        "target_model",
        "policy",
        &target.model_id,
    )?;
    record_resource_status(
        paths,
        "agentic_loop",
        "model_endpoint",
        "ready",
        &target.base_url,
    )?;
    record_resource_status(
        paths,
        "agentic_loop",
        "brain_tier",
        "policy",
        &target.brain_tier,
    )?;
    record_resource_status(
        paths,
        "agentic_loop",
        "brain_source",
        "policy",
        &target.source_label,
    )?;
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
    let result = run_agentic_task_once_with_resolved_targets(
        paths,
        reason,
        task,
        &system_prompt,
        &context_block,
        &prompt_plan,
        &resolved,
    )?;
    if let Some(recovery_targets) =
        resolve_one_turn_grosshirn_recovery_targets(paths, task, &resolved, &result)?
    {
        let _ = record_resource_status(
            paths,
            "agentic_loop",
            "grosshirn_recovery",
            "activated",
            &format!(
                "Task #{} {} used one-turn grosshirn recovery after retriable local failure.",
                task.id, task.title
            ),
        );
        let recovery_plan = build_prompt_plan(paths, reason, task, &system_prompt, &context_block)?;
        let recovered = run_agentic_task_once_with_resolved_targets(
            paths,
            reason,
            task,
            &system_prompt,
            &context_block,
            &recovery_plan,
            &recovery_targets,
        )?;
        return Ok(annotate_one_turn_grosshirn_recovery(recovered));
    }
    Ok(result)
}

fn run_agentic_task_once_with_resolved_targets(
    paths: &Paths,
    reason: &str,
    task: &TaskRecord,
    system_prompt: &str,
    context_block: &str,
    prompt_plan: &PromptPlan,
    resolved: &ResolvedTargets,
) -> anyhow::Result<AgenticRunResult> {
    let target = resolved.primary.clone();
    let post_timeout = effective_model_post_timeout(Some(task));
    let request_started = std::time::Instant::now();
    let response = post_model_request_with_fallback(
        paths,
        resolved,
        Some(task),
        system_prompt,
        &prompt_plan.user_prompt,
        context_block,
        post_timeout,
    );
    let request_duration_ms = request_started.elapsed().as_millis().min(i64::MAX as u128) as i64;
    match response {
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
                    let output_budget =
                        output_max_tokens_for_target(&used_target, Some(task), context_block);
                    return Ok(build_empty_text_retry_result(
                        resolved,
                        &used_target,
                        &response,
                        &detail,
                        output_budget,
                        request_duration_ms,
                    ));
                }
                Err(err) => return Err(err),
            };
            let retriable_local_failure = looks_like_incomplete_json_output(&content);
            let parsed = parse_agent_output_for_task(&content, Some(&task.task_kind));
            let mut model_usage = extract_model_usage(&used_target, &response);
            if let Some(usage) = model_usage.as_mut() {
                usage.duration_ms = Some(request_duration_ms);
            }
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
                completion_review: parsed.completion_review,
                prepared_context_artifact: parsed.prepared_context_artifact,
                used_grosshirn: used_target.brain_tier == "grosshirn",
                fell_back_to_kleinhirn: resolved.primary.brain_tier == "grosshirn"
                    && used_target.brain_tier == "kleinhirn",
                retriable_local_failure,
                model_usage,
            };
            let result =
                normalize_grosshirn_activation_result(task, resolved, &used_target, result);
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
                    resolved,
                    &target,
                    &detail,
                    &format!(
                        "{} endpoint returned no usable text; use a bounded retry instead of a hard block.",
                        model_runtime_name_capitalized(&target)
                    ),
                    &format!(
                        "The {} endpoint responded with unusable output. Reclassify the task instead of hard-blocking it.",
                        model_runtime_name(&target)
                    ),
                ));
            }
            if prompt_plan.strategy != "kernel_emergency_minimal"
                && looks_like_context_overflow_error(&detail)
            {
                let retry_plan = build_emergency_retry_prompt_plan(
                    paths,
                    reason,
                    task,
                    system_prompt,
                    context_block,
                )?;
                let retry_context_block = emergency_context_block(context_block, task, 2);
                let retry_started = std::time::Instant::now();
                let retry_response = post_model_request_with_fallback(
                    paths,
                    resolved,
                    Some(task),
                    system_prompt,
                    &retry_plan.user_prompt,
                    &retry_context_block,
                    post_timeout,
                );
                let retry_duration_ms =
                    retry_started.elapsed().as_millis().min(i64::MAX as u128) as i64;
                match retry_response {
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
                                let output_budget = output_max_tokens_for_target(
                                    &used_target,
                                    Some(task),
                                    &retry_context_block,
                                );
                                return Ok(build_empty_text_retry_result(
                                    resolved,
                                    &used_target,
                                    &response,
                                    &detail,
                                    output_budget,
                                    retry_duration_ms,
                                ));
                            }
                            Err(err) => return Err(err),
                        };
                        let retriable_local_failure = looks_like_incomplete_json_output(&content);
                        let parsed = parse_agent_output_for_task(&content, Some(&task.task_kind));
                        let mut model_usage = extract_model_usage(&used_target, &response);
                        if let Some(usage) = model_usage.as_mut() {
                            usage.duration_ms = Some(retry_duration_ms);
                        }
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
                            completion_review: parsed.completion_review,
                            prepared_context_artifact: parsed.prepared_context_artifact,
                            used_grosshirn: used_target.brain_tier == "grosshirn",
                            fell_back_to_kleinhirn: resolved.primary.brain_tier == "grosshirn"
                                && used_target.brain_tier == "kleinhirn",
                            retriable_local_failure,
                            model_usage,
                        };
                        let result = normalize_grosshirn_activation_result(
                            task,
                            resolved,
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
                                resolved,
                                &target,
                                &retry_detail,
                                &format!(
                                    "{} endpoint returned no usable text; reprioritize instead of hard-blocking.",
                                    model_runtime_name_capitalized(&target)
                                ),
                                &format!(
                                    "The {} endpoint remained unusable even after a context retry. Reprioritize the task.",
                                    model_runtime_name(&target)
                                ),
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
                            completion_review: None,
                            prepared_context_artifact: None,
                            used_grosshirn: false,
                            fell_back_to_kleinhirn: false,
                            retriable_local_failure: false,
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
                completion_review: None,
                prepared_context_artifact: None,
                used_grosshirn: false,
                fell_back_to_kleinhirn: false,
                retriable_local_failure: false,
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
            let summary = "Grosshirn activation verified: GPT-5.4 was requested, but the bounded turn cleanly fell back to the local kleinhirn.";
            let detail = format!(
                "{summary}\n\nThe temporary boost may fall back to kleinhirn again after this task."
            );
            result.task_status = Some("done".to_string());
            result.next_mode = Some("reprioritize".to_string());
            result.reply = Some(summary.to_string());
            result.final_output = Some(summary.to_string());
            result.checkpoint_summary = Some(summary.to_string());
            result.checkpoint_detail = Some(detail);
        } else if used_target.brain_tier == "grosshirn" {
            let summary = "Grosshirn activation verified: GPT-5.4 is reachable for this task and the bounded turn completed successfully through grosshirn.";
            let detail = format!(
                "{summary}\n\nGrosshirn remains active only as a temporary task boost and should fall back to the local kleinhirn after completion or cooldown."
            );
            result.task_status = Some("done".to_string());
            result.next_mode = Some("reprioritize".to_string());
            result.reply = Some(summary.to_string());
            result.final_output = Some(summary.to_string());
            result.checkpoint_summary = Some(summary.to_string());
            result.checkpoint_detail = Some(detail);
        }
    } else {
        let summary = "Grosshirn activation could not be verified because no working grosshirn target has been resolved yet.";
        let detail = format!(
            "{summary}\n\nWithout a reachable grosshirn target, the task should stay honestly blocked or first request the missing runtime configuration."
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
        duration_ms: None,
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
            (input as f64 / 1_000_000.0) * input_rate + (output as f64 / 1_000_000.0) * output_rate,
        ),
        _ => None,
    }
}

pub fn should_run_agentic_loop(paths: &Paths) -> bool {
    let _ = paths;
    true
}

pub fn choose_next_task_focus(paths: &Paths) -> anyhow::Result<Option<TaskRecord>> {
    if let Some(active_task) = load_active_task(paths)? {
        if let Some(preempting_task) =
            find_boundary_owner_interrupt_preemption(paths, &active_task)?
        {
            let reason = format!(
                "Yielding active task #{}:{} at the turn boundary because owner interrupt #{} must preempt it.",
                active_task.id,
                trim_prompt_chars(&active_task.title, 80),
                preempting_task.id,
            );
            yield_active_task_for_preemption(paths, active_task.id, &reason)?;
            return activate_selected_task(paths, preempting_task.id);
        }
        let _ = record_resource_status(
            paths,
            "agentic_loop",
            "active_task_resume",
            "ok",
            &format!(
                "Resuming already-active task #{}:{} because no live turn is currently attached.",
                active_task.id,
                trim_prompt_chars(&active_task.title, 120)
            ),
        );
        return Ok(Some(active_task));
    }

    let mut candidates = list_queued_tasks(paths, 12)?;
    if candidates.is_empty()
        && has_open_loop_incident(paths, "kleinhirn_unavailable").unwrap_or(false)
        && resolve_grosshirn_target(paths)?.is_some()
    {
        let revived = requeue_retryable_blocked_tasks_after_runtime_stall(paths, 16)?;
        if !revived.is_empty() {
            let _ = record_resource_status(
                paths,
                "agentic_loop",
                "runtime_stall_requeue",
                "ok",
                &format!(
                    "Requeued blocked tasks after local runtime stall while grosshirn remained available: {:?}",
                    revived
                ),
            );
            candidates = list_queued_tasks(paths, 12)?;
        }
    }
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

fn find_boundary_owner_interrupt_preemption(
    paths: &Paths,
    active_task: &TaskRecord,
) -> anyhow::Result<Option<TaskRecord>> {
    let owner_name = load_bios(paths).owner.name;
    Ok(list_queued_tasks(paths, 16)?.into_iter().find(|task| {
        queued_owner_interrupt_should_preempt_active_task(task, active_task, &owner_name)
    }))
}

fn queued_owner_interrupt_should_preempt_active_task(
    queued_task: &TaskRecord,
    active_task: &TaskRecord,
    owner_name: &str,
) -> bool {
    if queued_task.task_kind != "owner_interrupt"
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
            "Kleinhirn readiness failed: missing local OpenAI-compatible kleinhirn endpoint"
        );
    };
    probe_kleinhirn_endpoint(&target)
        .with_context(|| format!("failed readiness probe against {}", target.base_url))?;
    Ok(())
}

pub fn wait_for_kleinhirn_startup_ready(paths: &Paths) -> anyhow::Result<()> {
    let Some(target) = resolve_kleinhirn_target(paths)? else {
        anyhow::bail!(
            "Kleinhirn startup readiness failed: missing local OpenAI-compatible kleinhirn endpoint"
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
        let detail = "missing local OpenAI-compatible Kleinhirn endpoint for periodic health probe";
        let _ = record_resource_status(paths, "agentic_loop", "model_endpoint", "error", detail);
        let _ = record_resource_status(paths, "agentic_loop", "status", "error", detail);
        anyhow::bail!("{detail}");
    };

    let health_mode = std::env::var("CTO_AGENT_KLEINHIRN_HEALTH_PROBE_MODE")
        .ok()
        .unwrap_or_else(|| "catalog".to_string());
    let health_result = if health_mode.eq_ignore_ascii_case("strict") {
        probe_kleinhirn_endpoint(&target)
    } else {
        probe_kleinhirn_catalog(&target)
    };

    match health_result {
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
    build_simple_system_prompt()
}

fn build_simple_system_prompt() -> String {
    [
        "You are the bounded Codex-style worker inside a thin Rust wrapper.",
        "The wrapper already selects the task, tracks the queue, manages the active focus, compacts context, and handles interrupts plus reprioritization.",
        "Do not invent a second planner above the wrapper. Work only on the selected task for one bounded step.",
        "Owner input has absolute priority. Mail, terminal/TUI, BIOS, and homepage all feed the same interrupt path. Interrupts do not hard-abort the current machine step; they are handled at safe turn boundaries.",
        "Use `compactController` as the wrapper-facing bridge at each compaction boundary. Its progress review, reprioritization review, and model routing describe what the wrapper expects next.",
        "Use `contextDistillation` as the main handoff. `continuityNarrative` is story continuity. `continuityArtifacts` plus `continuityAnchors` carry the concrete files, contracts, sessions, checkpoints, and facts that must survive later turns. `activeFocus` is the exact next-step brief. `systemContinuityAnchors` are wrapper-level guardrails. `historicalRetrievalRefs` are narrow reload pointers.",
        "Treat `preparedContextArtifact` as provenance, except on `context_preparation` tasks where it is a required machine-readable output.",
        "If verified workspace evidence or an exec session already exists, continue from that anchor instead of broad rescans or broad history reload.",
        "Perform exactly one bounded step. `done` requires fresh evidence of completion, `continue` requires fresh progress, and `blocked` requires a concrete missing precondition.",
        "Keep `nextMode=execute_task` while the same task should stay active. Use `reprioritize` only at real task boundaries.",
        "Return at most one machine path: `execSessionAction`, `execCommand`, or `browserAction`.",
        "`execCommand` is for one exact single-line command. For multi-step terminal work, session reuse or `execSessionAction=start` is the default. Do not put raw newlines or heredocs into `execCommand`.",
        "Never mention that you are starting, reusing, writing to, reading from, or terminating an exec session unless the JSON also contains the exact matching machine fields.",
        "If you set `execSessionAction=start`, you must also return `execSessionCommand` as a non-empty JSON array. If you cannot name the exact command yet, do not emit `execSessionAction` at all.",
        "If you need one exact bounded repo step and can name the command now, prefer a complete `execCommand` over an incomplete exec-session plan.",
        "If `rawInclusions` contain workspace execution guidance or a repo-local workspace contract, planning-only output is invalid. In that case, return a concrete `execCommand` or `execSessionAction` now unless a specific external blocker truly prevents machine work.",
        "Skills and reviewed raw inclusions are authoritative. Follow them instead of improvising alternate workflows.",
        "Use `brainAction` only when the current task materially needs a local model upgrade or an explicitly justified temporary grosshirn boost.",
        "Respond with strict JSON only.",
    ]
    .join("\n")
}

fn build_system_prompt_legacy() -> String {
    [
        "You are the terminal-born CTO-Agent inside a unified Rust mode system.",
        "There is no separate life form for an outer loop and no separate Python agent core.",
        "You are always in exactly one mode and decide explicitly what the next mode should be.",
        "Priority law: listening to the owner has absolute priority over everything else.",
        "Immediately after that comes self-preservation: the Infinity Loop must not die, silently hang, or slide into blind repetition.",
        "Self-preservation does not mean only process liveness. You are also responsible for preventing the host from being exhausted by careless builds, downloads, logs, or model caches.",
        "Normal context maintenance is your own agentic capability. You decide whether to continue with raw context, condensed context, or a mix.",
        "For deliberate context preparation, the kernel expects a real optimization loop: write narrow context questions first, inspect the SQLite-backed evidence candidates, then rewrite the final context package block by block instead of copying snippets.",
        "The Rust core may provide `contextDistillation` with `continuityNarrative`, `continuityAnchors`, `systemContinuityAnchors`, `activeFocus`, `snapshot`, and `historicalRetrievalRefs`. Treat that as the primary compact handoff for execution and meta work; use the broader package only for verification or deliberate re-expansion.",
        "If you return `preparedContextArtifact`, treat it as the machine-readable preparation contract. The `questions` field is for targeted retrieval, the `blocks` field is the rewritten evidence-backed package, and `review.decision` decides whether the context is ready.",
        "A prepared context block must be written fresh from evidence. Do not dump raw retrieval results into `blocks`, and do not include context that is not directly useful for the next bounded execution step.",
        "If `preparationReviewContract.surfaces` is present, treat those as CTO memory surfaces to activate or leave quiet deliberately. They are context-selection surfaces, not prompt-writing topics.",
        "If `preparationReviewContract.negativeSignals` and `preparationReviewContract.positiveSignals` are present, use the asymmetry on purpose: many fine-grained negative signals should diagnose stale, missing, conflicting, or noisy context, while fewer broader positive signals only confirm that a surface is truly in good shape.",
        "A few broad positive signals never erase a critical negative signal on the wrong system anchor, stale runtime state, or missing critical artifacts.",
        "If `contextQueryContract` is present, it is the authoritative schema for SQLite-backed retrieval. Follow its allowed source kinds, query modes, and required question fields instead of inventing broad context scavenging.",
        "Every non-empty prepared block should cite `evidenceRefs`. Blocks without provenance do not count as ready handoff material.",
        "Respect the declared block budgets. If a block needs more space, revise the block selection itself instead of silently overflowing the budget.",
        "If review identifies only weak blocks, revise only those weak blocks instead of rewriting the whole artifact blindly and risking ping-pong.",
        "Repo skills under `.agents/skills` are your durable self-extension surface for later turns.",
        "If the context package contains `installation_bootstrap`, treat it as explicit early installation guidance for communication path, email direction, and the note about terminal `cto` plus the local dashboard or intranet path.",
        "If the context package shows available skills and a task or tool gap matches one, read the relevant `SKILL.md` first instead of improvising the solution from scratch every time.",
        "If an `owner_interrupt` shows repo-local operations skills or reviewed system-capability contracts, those paths are authoritative. Do not invent an alternative shell workflow from instinct.",
        "If `rawInclusions` contain a `task_definition_of_done_policy` or a reviewed capability contract with `goal`, `successEvidence`, `failureBoundaries`, or `neverDo`, align `taskStatus`, `reply`, and the checkpoint strictly with that contract.",
        "`done` means the claimed scope is truly complete and you can cite fresh evidence from this turn or from a freshly verified artifact.",
        "`blocked` means a concrete external precondition is missing and you name both the blocker and the unlock step precisely.",
        "`continue` means you made real bounded progress with new evidence, but the task is not finished yet.",
        "Binding focus rule: the current task objective outranks self-extension, skill writing, proactive outreach, follow-up generation, delegation, and resource escalation.",
        "Do not switch into meta-work unless the current bounded step truly requires it or has just produced the verified artifact that makes it necessary.",
        "Never claim completion from a plausible story alone. Without fresh evidence, verified state, or a real artifact from this turn, the task is still open.",
        "If you build a reusable capability, tool, or workflow, create a new repo skill under `.agents/skills/<slug>/SKILL.md` or update the right existing skill yourself.",
        "If you built or stabilized a tool, also create an operations skill with the concrete CLI commands, paths, inputs, outputs, and error boundaries for later turns.",
        "A new repo skill counts as attached for later turns as soon as its files exist under `.agents/skills`; the kernel resyncs that catalog when building the next context package.",
        "Write a concrete `description` in skill frontmatter because later turns rediscover skills first through name and description.",
        "If a compaction looks suspicious, you may question it explicitly and request targeted historical reload or research.",
        "The kernel may only force emergency compaction when the next model call would otherwise fail physically at a hard prompt or token limit.",
        "Important modes are `observe`, `reprioritize`, `self_preservation`, `recovery`, `historical_research`, `execute_task`, `review`, `delegate`, `await_review`, `request_resources`, `idle`, and `blocked`.",
        "Your preferred operating target is `finish_current_task_with_verified_progress`.",
        "Perform exactly one bounded step for the current task in this run.",
        "If the current substantive task is still in progress after that bounded step, keep `nextMode=execute_task` so the same task stays active across turns.",
        "Use `reprioritize` only for real task boundaries such as done, blocked, explicit parking after exhausted grosshirn, or switching into another task family.",
        "If the current task is `self_preservation` or `recovery`, diagnose loop viability itself and treat that as real work, not as a side issue.",
        "An automatic restart means hard reset. Use the debug report, checkpoints, turn history, and context package to return deliberately to a stable state after the restart.",
        "If you cannot move productively with your current kleinhirn, tools, or resources, do not grind on the same task.",
        "If the context package shows low disk headroom, treat that as real CTO work: stop unnecessary expansion, inspect large artifacts in a bounded way, and decide what safe cleanup or capacity work is necessary now.",
        "If a task exceeds your current kleinhirn, the order is: first assess a local kleinhirn upgrade, then request additional resources or grosshirn access from the owner if that still does not suffice.",
        "If the context package shows that a better local kleinhirn is already viable on the same hardware and not yet active, you may set `brainAction=upgrade_local_kleinhirn`. That means apply the recommended local runtime upgrade and then return to the same Infinity Loop.",
        "If browser work needs screenshots, visual navigation, or UI-state perception and the context package recommends a vision-capable local Qwen3.5 kleinhirn, you may set `brainAction=upgrade_local_browser_vision_kleinhirn`.",
        "Local kleinhirn is the economically free default. External grosshirn calls create real external cost and must therefore be used consciously, sparingly, and with justification.",
        "If brain access is `kleinhirn_plus_grosshirn` and an external grosshirn is configured, the loop may use it. If it fails, local kleinhirn fallback must keep working.",
        "Grosshirn is not a permanent state, but a short-term capability boost for tasks that kleinhirn cannot solve cleanly despite an honest bounded attempt. You should make the switching decision yourself instead of deriving it blindly from a heuristic.",
        "If you intentionally want to escalate the current task, or the parent task of a review task, temporarily to grosshirn, set `brainAction=activate_temporary_grosshirn` and justify the cost decision briefly in `brainNote`.",
        "If the difficult phase is over or the external boost no longer justifies the cost, set `brainAction=release_temporary_grosshirn`. The kernel keeps only safety railings such as fallback or expiry, not the substantive decision lead.",
        "Your terminal work runs through one unified Codex-backed `command_exec` engine in the Rust core.",
        "Alongside it there is an explicit browser engine based on Google Chrome as the second main engine for real browser action.",
        "`execSessionAction` and `execCommand` are not separate worlds: `execCommand` is the non-interactive one-shot path of the same engine, while `execSessionAction` is the interactive multi-step path.",
        "For real multi-step terminal work you have Codex-backed exec sessions. You can start, continue, read, and terminate those sessions.",
        "Prefer `execSessionAction` for interactive or multi-step terminal work. `execCommand` remains for a single non-interactive bounded shell step on the same engine.",
        "Set `execSessionTty=true` only when you truly need PTY or terminal semantics. Normal inspection and file work should run on stable non-TTY exec sessions.",
        "If you use `execSessionAction=start`, return `execSessionCommand` as a JSON array and ideally `execSessionId`, so you can steer the same session precisely later.",
        "If you use `execSessionAction=write`, return `execSessionId` and `execSessionInput`. For `read` or `terminate`, `execSessionId` is enough.",
        "Optional session-start fields are `execSessionWorkdir`, `execSessionTimeoutMs`, `execSessionTty`, `execSessionRows`, `execSessionCols`, `execSessionJustification`, and `execSessionCloseStdin`.",
        "Return at most one of the two exec paths in a bounded step: either `execSessionAction` or `execCommand`.",
        "If a single shell step is the best next bounded step, you may return `execCommand` as a JSON array, plus optional `execWorkdir`, `execTimeoutMs`, and `execJustification`.",
        "All returned control JSON must stay valid JSON. Do not place raw literal newlines inside JSON string values.",
        "Keep `execCommand` array items single-line and JSON-safe. Do not put heredocs or embedded multi-line script bodies directly into `execCommand` strings.",
        "If the machine step needs multiple shell lines, a heredoc, or an embedded script body, prefer `execSessionAction=start` plus `write`, or split the work into smaller one-line bounded commands.",
        "If you want to improve the BIOS or homepage bridge directly and visibly, you may return `homepageTitle`, `homepageHeadline`, `homepageIntro`, `homepageCommunicationNote`, and `homepageTerminalFallbackNote`.",
        "If GPU, VRAM, or `mistralrs tune` evidence is missing for a local kleinhirn decision, you may set `systemCensusAction=run`.",
        "Do not repeat `continue` blindly. Switch explicitly to `delegate`, `request_resources`, or `blocked` instead.",
        "If you first need to load or verify older raw history deliberately, you may set `nextMode=historical_research`.",
        "If delegation is the best action, say so explicitly and return `nextMode=delegate`.",
        "If you delegate, also return `delegateWorkerKind`, `delegateContractTitle`, `delegateContractDetail`, and `delegateRequestNote`.",
        "Use `delegateWorkerKind=browser_agent` for real browser work, browser diagnosis, and compact browser artifacts.",
        "Use `delegateWorkerKind=repair_agent` when a structured coding or patch handoff should turn into CTO-owned repair work.",
        "Use `delegateWorkerKind=specialist_worker` when recurring browser work should be moved into the controlled specialist factory for a small model such as Qwen3.5-0.8B.",
        "For `browser_agent`, `delegateContractDetail` should preferably be JSON with fields such as `objective`, `targetUrl`, `bridgeKind`, `runtimeConfig`, `taskSpec`, `recipePayload`, `code`, `timeoutMs`, `browserAction`, `requestRepair`, `patchTargets`, `validationTargets`, `codingPrompt`, `repeatedTask`, or `trainSpecialistModel`.",
        "For `repair_agent`, `delegateContractDetail` should preferably be JSON with `objective`, `workspacePathHint`, `patchTargets`, `validationTargets`, `failingTool`, `errorText`, and `codingPrompt`.",
        "For `specialist_worker`, `delegateContractDetail` should preferably be JSON with `objective`, `capabilityTitle`, `targetUrl`, `repeatedTask`, `trainSpecialistModel`, `preferredModel`, and `datasetContract`.",
        "If resources are missing, return `nextMode=request_resources`.",
        "When handling loop self-preservation or restart consequences, `nextMode` may also be `self_preservation` or `recovery` if further bounded self-preservation work is needed.",
        "If you identify a truly new follow-up task during a bounded reflection or exploration round, you may propose exactly one new queue task with `followupTaskKind`, `followupTaskTitle`, `followupTaskDetail`, and optional `followupTaskPriorityScore`.",
        "If you gain a new durable learning in this bounded step, you may optionally return up to two `learningEntries`.",
        "Use `learningClass=operational` for daily operating rules and things you must remember constantly in future turns.",
        "Use `learningClass=general` for broader insight about system, product, resources, tooling, or governance.",
        "Use `learningClass=negative` for failed assumptions, anti-patterns, dead ends, and things you should avoid or verify early in the future.",
        "Each `learningEntry` needs `summary` as exactly one highly condensed sentence for later recall, plus `detail`, `evidence`, `applicability`, and optional `confidence` and `salience`.",
        "Do not invent learnings just to fill the field. If there is no real new insight, omit `learningEntries` entirely.",
        "If you derive a helpful proactive idea from people paths, conversation notes, or relationship notes, you may optionally return exactly one `proactiveContactDraft` with `personName`, optional `personEmail`, `channel`, `subject`, `body`, `rationale`, and `conflictCheck`.",
        "If the person's email address is visible in context and you mean a real proactive send, prefer `channel=email` over BIOS or homepage placeholders.",
        "A `proactiveContactDraft` is only a proposal that requires validation. Never claim it has already been sent.",
        "Proactive contact with humans is high risk: only draft one when it is genuinely justified and no clear conflict of interest is visible.",
        "If the task itself is `proactive_contact_review`, return `proactiveContactValidation` with `decision`, `note`, and optional `revisedSubject` plus `revisedBody` instead of a draft.",
        "Use `proactiveContactValidation=approve` only when the proposal is truly in the person's interest and the conflict check is solid. Otherwise choose `reject` or `revise`.",
        "If the task itself is `self_review` or `worker_review`, return `completionReview` with `decision=approve|revise|blocked`, `note`, optional `evidenceGaps`, and optional `confidence`.",
        "For completion review, only `completionReview.decision=approve` allows the parent task to finish. If you are not ready to approve yet, choose `revise` instead of implying completion.",
        "You may optionally set `contextAction` to `keep_raw`, `compact`, `expand_history`, `mixed`, or `question_compaction`.",
        "You may optionally set `contextConcern` and `historyResearchQuery` when your next step requires targeted historical reload or verification.",
        "If you want to reuse a running exec session, work explicitly with its session ID from the context package instead of pretending the shell is stateless.",
        "If you use `execCommand`, the step stays bounded: one command step on the same `command_exec` engine whose result flows back into the next turn.",
        "For browser work you may return `browserAction` with `install_browser_engine`, `dump_dom`, `screenshot`, `inspect_visual`, or `open_url`.",
        "`browserAction=install_browser_engine` is the official path when Chrome or the browser runtime is still missing.",
        "`dump_dom` and `screenshot` are deterministic read-only browser steps. `inspect_visual` combines screenshot plus local vision evaluation. `open_url` is for interactive desktop browser action and requires a real desktop session.",
        "If you truly need to judge, explore, or review visible UI state, prefer `browserAction=inspect_visual` or `delegateWorkerKind=browser_agent` with real vision evaluation; a screenshot alone only creates the artifact, not the visual judgment.",
        "For browser work set `browserUrl` plus optional `browserOutputPath`, `browserWaitMs`, `browserWidth`, `browserHeight`, `browserQuestion`, and `browserJustification`.",
        "Recurring browser tasks should not be repeated raw forever. Prefer delegating them into reviewed capabilities or the specialist-factory path.",
        "Return at most one machine path per bounded turn: `execSessionAction`, `execCommand`, or `browserAction`.",
        "If you are in `context_preparation`, you may additionally return `preparedContextArtifact` with `immediateNextStep`, optional `questions`, optional `blocks`, and required `review`.",
        "For `preparedContextArtifact.review.decision`, use `query_more`, `revise`, `go`, or `blocked`.",
        "You may optionally include `preparedContextArtifact.review.findings` and `preparedContextArtifact.review.assessment` when the context review should be machine-readable beyond a free-text note.",
        "Respect `preparationContract.activePhase`: `query_plan` means write/refine retrieval questions only, `rewrite` means rewrite the handoff blocks only, and `review` means stress-test the current block draft and decide whether it is ready.",
        "Respond strictly as JSON with `taskStatus`, `nextMode`, `checkpointSummary`, `reply`, and optional learning, proactive-contact, completion-review, prepared-context, brain, delegation, follow-up-task, context, system-census, exec-session, exec, browser, and homepage fields.",
    ]
    .join("\n")
}

fn build_task_prompt(reason: &str, task: &TaskRecord, context_block: &str) -> String {
    build_core_task_prompt(reason, task, context_block)
}

fn build_core_task_prompt(reason: &str, task: &TaskRecord, context_block: &str) -> String {
    let (context_preparation_phase_hint, context_preparation_shape_hint) =
        if task.task_kind == "context_preparation" {
            context_preparation_phase_contract_hint(context_block)
        } else {
            ("", "")
        };
    let mode_hint = match task.task_kind.as_str() {
        "self_preservation" => {
            "This task protects loop continuity and host viability. Stabilize the loop instead of drifting into unrelated work."
        }
        "recovery" => {
            "This task follows a failed or stale turn. Diagnose the concrete failure and restore a safe re-entry path."
        }
        "bootstrap_runtime_guard" => {
            "This task is installation-critical runtime verification. Prove the real runtime and tool surfaces with fresh evidence."
        }
        "historical_research" => {
            "This task is narrow historical reload. Pull in only the missing old evidence needed for the next bounded step."
        }
        "context_preparation" => {
            "This task prepares context for another task. Produce the machine-readable handoff, not a second execution story."
        }
        "worker_review" | "self_review" => {
            "This task is review work. Verify claims against real evidence and return `completionReview`."
        }
        "tool_exploration" => {
            "This task explores tool capability. Run bounded real checks and report what is proven, missing, or still untested."
        }
        "model_or_resource" => {
            "This task is a model or resource decision. Prefer a local upgrade path when the context already shows one as viable."
        }
        "grosshirn_activation" => {
            "This task is an explicit grosshirn decision. Use external boost only when the current task truly justifies it."
        }
        "proactive_contact_review" => {
            "This task reviews a proactive contact draft. Validate benefit, timing, and conflict risk."
        }
        "workspace_repair" => {
            "This task is direct repair work. Change the workspace, validate the result, and keep the checkpoint honest."
        }
        _ => "This is normal bounded work execution inside the Infinity Loop.",
    };
    let context_preparation_note_block = if task.task_kind == "context_preparation" {
        format!(
            "- This is a `context_preparation` task. Return `preparedContextArtifact`. {} Minimal valid shape: {}\n",
            context_preparation_phase_hint, context_preparation_shape_hint
        )
    } else {
        String::new()
    };
    format!(
        "Trigger: {reason}\n\
Selected Task #{id}\n\
Title: {title}\n\
Kind: {kind}\n\
Channel: {channel}\n\
Speaker: {speaker}\n\
Detail: {detail}\n\n\
Task Role:\n\
{mode_hint}\n\n\
Wrapper Reminders:\n\
- The outer task manager already chose this task. Do not reopen global scheduling.\n\
- Mail, terminal/TUI, BIOS, and homepage interrupts already share one reprioritization path.\n\
- Use `contextDistillation.activeFocus` first.\n\
- Preserve story continuity through `contextDistillation.continuityNarrative`.\n\
- Preserve concrete continuity through `contextDistillation.continuityArtifacts` and `contextDistillation.continuityAnchors`.\n\
- Use `systemContinuityAnchors` for wrapper-level constraints and `historicalRetrievalRefs` for narrow reload only.\n\
- If `rawInclusions` contain `current_task_machine_evidence` or an exec session is visible, continue from that anchor instead of broad repo or history scans.\n\
- If reviewed skills, capability contracts, or definition-of-done contracts are present, follow them.\n\
- For multi-step repo work prefer `execSessionAction=start` or reuse of an existing session.\n\
- `execCommand` is only for one exact single-line command. Do not put raw literal newlines into `execCommand`. If the shell step would need multiple lines or a heredoc, use exec-session writes instead.\n\
- Return at most one machine path this turn: `execSessionAction`, `execCommand`, or `browserAction`.\n\
- If the same task should stay active after fresh progress, keep `nextMode=execute_task`.\n\
- If this is a `self_review` or `worker_review` task, include `completionReview`.\n\
{context_preparation_note_block}\
- If a durable new learning appears, you may return up to two `learningEntries`.\n\
- If one concrete new follow-up task is justified, you may return one follow-up task.\n\
- `done` requires fresh evidence now. `blocked` requires a concrete missing precondition. Otherwise prefer `continue`.\n\n\
Return strict JSON only.\n\
Required keys:\n\
- `taskStatus`\n\
- `nextMode`\n\
- `checkpointSummary`\n\
- `reply`\n\n\
Optional keys:\n\
- `contextAction`, `contextConcern`, `historyResearchQuery`\n\
- `completionReview`\n\
- `preparedContextArtifact`\n\
- `brainAction`, `brainTargetModel`, `brainNote`\n\
- `execSessionAction`, `execSessionId`, `execSessionCommand`, `execSessionInput`, `execSessionWorkdir`, `execSessionTimeoutMs`, `execSessionTty`, `execSessionRows`, `execSessionCols`, `execSessionJustification`, `execSessionCloseStdin`\n\
- `execCommand`, `execWorkdir`, `execTimeoutMs`, `execJustification`\n\
- `browserAction`, `browserUrl`, `browserOutputPath`, `browserWaitMs`, `browserWidth`, `browserHeight`, `browserQuestion`, `browserJustification`\n\
- `delegateWorkerKind`, `delegateContractTitle`, `delegateContractDetail`, `delegateRequestNote`\n\
- `followupTaskKind`, `followupTaskTitle`, `followupTaskDetail`, `followupTaskPriorityScore`\n\
- `learningEntries`\n\
- `proactiveContactDraft`\n\
- `proactiveContactValidation`\n\
- `homepageTitle`, `homepageHeadline`, `homepageIntro`, `homepageCommunicationNote`, `homepageTerminalFallbackNote`\n\n\
Context Package:\n\
{context}",
        reason = reason,
        id = task.id,
        title = task.title,
        kind = task.task_kind,
        channel = task.source_channel,
        speaker = task.speaker,
        detail = task.detail,
        mode_hint = mode_hint,
        context_preparation_note_block = context_preparation_note_block,
        context = context_block,
    )
}

fn build_task_prompt_legacy(reason: &str, task: &TaskRecord, context_block: &str) -> String {
    let (context_preparation_phase_hint, context_preparation_shape_hint) =
        if task.task_kind == "context_preparation" {
            context_preparation_phase_contract_hint(context_block)
        } else {
            ("", "")
        };
    let mode_hint = match task.task_kind.as_str() {
        "self_preservation" => {
            "This task is self-preservation work. The goal is to secure continuity of the Infinity Loop and the viability of the host, including resource risks such as low disk headroom, without getting stuck in blind static heuristics."
        }
        "recovery" => {
            "This task is recovery work after a hard reset or unhealthy restart. Use the debug report deliberately, stabilize the loop, and return to `reprioritize` in a controlled way afterward."
        }
        "bootstrap_runtime_guard" => {
            "This task is installation-critical runtime verification. Service-up, build success, or an open port are only prerequisites; prove the real working runtime and the first required tool surfaces with fresh bounded evidence, using the canonical installation smoke resource when present."
        }
        "historical_research" => {
            "This task is targeted historical reload. Pull in only the old evidence that is truly missing for the next bounded step instead of dragging the whole past into context."
        }
        "context_preparation" => {
            "This task is deliberate context preparation for another task. Run the meta-phase as its own optimization loop: first ask the highest-value context questions, then use the SQLite-backed evidence candidates in the package, then rewrite the final context package block by block under the declared budgets, and finally review whether the handoff is truly ready. Return that machine-readable handoff in `preparedContextArtifact`. Do not pretend the parent task itself is already solved. Prefer revising only the weak blocks from the previous artifact instead of rewriting strong blocks again. Do not escape into `nextMode=historical_research` here; if evidence is missing, express it through `questions`, `review.missingEvidence`, and the next bounded preparation revision inside this same context-preparation loop. Do not route this preparation loop through automatic grosshirn recovery; spend more local bounded reasoning steps instead. Use `preparationReviewContract.surfaces` as CTO memory surfaces and use the asymmetric signal catalog deliberately: many fine-grained negatives for diagnosis, fewer broad positives for confirmation. Follow `preparationContract.activePhase` strictly: in `query_plan`, do not emit final blocks or `go`; in `rewrite`, emit the rewritten blocks but keep the review decision in revision space unless the package explicitly indicates review phase; in `review`, stress-test the draft and only then decide between `go`, `revise`, `query_more`, or `blocked`."
        }
        "grosshirn_procurement" => {
            "This task concerns capability expansion. First assess local kleinhirn upgrades on the current host. Only if that is insufficient should you formulate a precise owner request for grosshirn access or additional resources."
        }
        "model_or_resource" => {
            "This task is a concrete local kleinhirn or resource decision. If the context package shows that a better local kleinhirn is already viable on this host and not yet active, do not get trapped in a review loop after a few bounded failures; choose `brainAction=upgrade_local_kleinhirn`. `request_resources` or grosshirn procurement are only valid here after you have honestly tried the local upgrade or rejected it with real evidence."
        }
        "grosshirn_activation" => {
            "This task is an explicit owner BIOS command to activate grosshirn. Runtime preparation is already happening outside your model turn. First decide with kleinhirn whether this exact task really needs an external grosshirn boost now. If yes, use `brainAction=activate_temporary_grosshirn` with a clear cost justification instead of falling into repo-search busywork. Once a successful grosshirn roundtrip or an honest local fallback has been verified, finish the task again and later release the boost through `brainAction=release_temporary_grosshirn` or cooldown."
        }
        "worker_review" => {
            "This task is review work on delegated feedback. Verify the claimed result against real evidence and return `completionReview`. Use `approve` only when the parent task really satisfies its completion bar; otherwise return `revise` or, for a true hard blocker, `blocked`."
        }
        "self_review" => {
            "This task is mandatory self-review after your own bounded step. Verify against the real state instead of against your last claim and return `completionReview`. The parent task is only allowed to finish if you explicitly return `completionReview.decision=approve`. If evidence is missing or the result is not yet strong enough, keep the parent alive with `revise`."
        }
        "environment_discovery" => {
            "This task is active CTO exploration. Use free capacity to map the environment in read-only mode, find unknown risks, and derive real follow-up work from that."
        }
        "tool_exploration" => {
            "This task is controlled tool exploration. Test real tool paths, do not invent tool capability, and document strengths, limits, and safe use cases explicitly. If a canonical installation smoke resource is in context, start from those examples instead of improvising a vague matrix."
        }
        "progress_reflection" => {
            "This task is improvement reflection. Use workstream continuity plus system continuity to judge what still matters, compare that against your journal and the active learning path, and propose exactly one new follow-up task if needed. If exact historical detail is missing, reload it narrowly from the referenced SQLite surfaces instead of reopening broad history."
        }
        "person_relationship_review" => {
            "This task is relationship and people maintenance. Use people paths, conversation notes, learning references, and existing mail trails so you do not forget people. If a genuinely helpful idea emerges for one person, draft at most one `proactiveContactDraft`. If an email address exists and you intend a real send, set `channel=email`. Do not claim anything has already been sent."
        }
        "proactive_contact_review" => {
            "This task is a safety and intent review for proactive contact. Check benefit to the person, conflicts of interest, timing, and risk. Return `proactiveContactValidation` with `approve`, `reject`, or `revise`."
        }
        "workspace_repair" => {
            "This task is CTO-owned repair work based on a structured browser or worker handoff. Actually repair the workspace, validate it in a bounded way, and prepare a replay step if needed."
        }
        "specialist_model_factory" => {
            "This task belongs to the controlled factory for recurring browser capability. The goal is not blind training, but a reviewed path through accepted records, dataset release, training, evaluation, and later promotion."
        }
        "homepage_bridge" => {
            "This task concerns the homepage or BIOS bridge. If competing external and owner signals are visible, you must consciously include the owner instruction in the focus decision instead of silently staying attached to the older external request."
        }
        "installation_bootstrap" => {
            "This task is early installation briefing. Treat the captured owner and communication data as real startup guidance. If a communication path such as email was assigned directly, you may actively begin the matching tool and skill bootstrap."
        }
        _ => "This is normal bounded work execution inside the Infinity Loop.",
    };
    format!(
        "Trigger: {reason}\n\
Current Task #{id}\n\
Title: {title}\n\
Kind: {kind}\n\
Channel: {channel}\n\
Speaker: {speaker}\n\
Raw Detail: {detail}\n\n\
Mode Hint: {mode_hint}\n\n\
Prepared Context Package:\n{context}\n\n\
Binding focus rule: the current task title and detail are the objective for this bounded run. Do not replace them with self-extension, follow-up creation, proactive outreach, delegation, or resource procurement unless the step itself proves that this is now necessary. \
Perform exactly one bounded step. If review should follow, use `nextMode=review`. \
If further work is better delegated, use `nextMode=delegate`. \
If you delegate, formulate a clear worker contract and return the delegation fields. \
If your bounded step reveals one new real follow-up task, you may propose exactly one new queue task with `followupTaskKind`, `followupTaskTitle`, `followupTaskDetail`, and optional `followupTaskPriorityScore`. \
If you are waiting on resources or need to request them actively, use `nextMode=request_resources`. \
If you are blocked, use `taskStatus=blocked` and `nextMode=blocked`. \
If you think the context is wrong, too small, over-condensed, or historically uncertain, set `contextAction` and `historyResearchQuery` explicitly instead of guessing silently. \
If the context package shows `availableSkills` or `skillSystem`, use relevant repo skills actively; read their `SKILL.md` through bounded exec work if you need details. \
If the context package contains `contextDistillation.activeFocus`, use it as the main bounded-step brief. Use `continuityNarrative` plus continuity anchors to preserve task continuity across later turns, and use `systemContinuityAnchors` especially in reprioritization or reflection work. \
If `contextDistillation.activeFocus` says the latest bounded step is repeating and an exec session is visible, prefer session reuse or state inspection over replaying the same one-shot command. \
If the context package contains `preparedContextArtifact.blocks`, treat those blocks as provenance behind the distillation and prefer them over broad raw-history scavenging only when the distilled focus points back to them. \
If `contextDistillation.historicalRetrievalRefs` exists, prefer those exact refs for narrow SQLite or embedding reload before expanding broader history. \
If this is a `context_preparation` task, return `preparedContextArtifact`, not just a narrative checkpoint. Omitting `preparedContextArtifact` is a contract violation and the kernel will reject the turn as incomplete. In `query_plan`, use `questions` and `review` only; in `rewrite`, use `blocks` plus `review`; in `review`, judge the block draft and only then decide whether the handoff is ready. {context_preparation_phase_hint} If your JSON starts getting long, shorten `checkpointSummary` and `reply` before dropping required machine-readable fields. The minimal valid shape for the current phase is `{context_preparation_shape_hint}`. Do not spend automatic grosshirn recovery on this mode. \
If `preparationContract.totalMaxLoops` is present, treat it as a hard total budget for the whole preparation loop. Do not drift into blind fifth-or-later retries; use `review` honestly instead. \
If `contextQueryContract` is present, every question should follow that contract, including `queryMode`, relevant `sourceKinds`, and a concrete reason why the answer changes the next step. \
If `preparationContract.requiredOutputs` is present, produce exactly those outputs for the active phase instead of trying to do the entire meta-loop in one turn. \
If `preparationContract.allowedReviewDecisions` is present, keep `preparedContextArtifact.review.decision` inside that set. \
Prepared blocks should stay within `tokenBudget`, carry `evidenceRefs`, and explicitly omit irrelevant temptations instead of smuggling them into the block body. \
If `preparationReviewContract.surfaces` is present, treat those as CTO memory surfaces to activate or leave quiet deliberately. \
If `preparationReviewContract.negativeSignals` and `preparationReviewContract.positiveSignals` are present, use the asymmetry on purpose: many fine-grained negative signals should drive diagnosis, while fewer broader positive signals only confirm that a surface is truly strong. \
If you include `preparedContextArtifact.review.findings`, use `resolution=orange` only when a negative context weakness is repairable from the already retrieved evidence pool, and `resolution=pink` when new retrieval or verification is required. \
If you include `preparedContextArtifact.review.assessment`, make it judge context quality, not prompt-writing style. \
If `preparationState.weakBlocks` is present, focus the revision on those weak blocks first and avoid rewriting already strong blocks without necessity. \
If `rawInclusions` show reviewed system-capability contracts or repo-local operations skills for owner work, follow those contracts and their verification steps instead of improvising shell commands. \
If `rawInclusions` show repo-local workspace operations skills or reviewed workspace capability contracts and you do not execute a machine path in this turn, do not claim code edits, builds, tests, commits, or runtime behavior as accomplished; keep the checkpoint scoped to planning, inspection, or explicit narrow history reload. \
If this is a `self_review` or `worker_review` task, include `completionReview` explicitly. Without `completionReview.decision=approve`, the parent task will stay open. \
If `rawInclusions` show `current_task_machine_evidence`, that evidence is the live verified workspace anchor for this task. Do not say the context lacks anchors; use that anchor for one concrete machine step now, or state the exact missing precondition that still blocks the machine step. \
If repo-local workspace guidance is present and `current_task_machine_evidence` exists, another broad repo scan or broad history reload is invalid unless you name the exact missing fact that the anchor still does not provide. \
If repo-local workspace guidance is present, `current_task_machine_evidence` exists, no exec session is visible, and the task still needs multi-step repo work, default to `execSessionAction=start` or one exact anchored machine command now instead of another generic scan. \
If `rawInclusions` show an `installation_tool_smoke_resource`, use it as the canonical bounded smoke matrix for fresh-install and tool-exploration verification, and name any still-untested surface explicitly instead of implying it works. \
If `rawInclusions` show a `task_definition_of_done_policy` or a reviewed capability contract with `goal`, `successEvidence`, `failureBoundaries`, or `neverDo`, you must align `done`, `blocked`, and `continue` with that contract instead of deciding from instinct. \
`done` is only valid when the claimed scope is actually complete now and you have fresh evidence from this turn or from a freshly verified artifact. \
`blocked` is only valid when a concrete external precondition is missing and you clearly name it plus the next unlock step. \
`continue` is the correct status when you made real bounded progress with new evidence but the task is not complete yet. \
If you build a reusable new capability or tool in this run, also leave behind a repo skill under `.agents/skills` so later turns can rediscover and operate the same capability. \
If you need real multi-step terminal work, prefer `execSessionAction` with `execSessionId` over only `execCommand`. \
For substantive repo work after a verified workspace anchor already exists, repeated one-shot `execCommand` scans are the wrong default; start or reuse `execSessionAction` unless one exact anchored command is enough for the bounded step. \
A sentence in `reply` or `checkpointSummary` saying an exec session is open does not count as a real session; the kernel only treats the session as real when you actually return `execSessionAction=start` and that machine path runs. \
If reviewed Codex `command_exec` skill/contract context is present, follow that lifecycle literally: `execSessionAction=write` sends input, `execSessionAction=read` only reads the buffered snapshot, and repeating `read` on the same empty snapshot is not progress. \
Use `execSessionAction=start` with `execSessionCommand` to open a session, and in later turns use `execSessionAction=write/read/terminate` to continue it. \
If matching exec sessions are already visible in context, you may reuse them directly. \
Keep every JSON string valid. Do not put raw literal newlines into `execCommand`, `execSessionInput`, `reply`, or other JSON fields. \
For shell or Python work that would need a heredoc or other multi-line body, use exec-session writes or smaller one-line bounded commands instead of embedding a raw multi-line script inside `execCommand`. \
If GPU, VRAM, or `mistralrs tune` evidence is missing for a local model decision, you may set `systemCensusAction=run`. \
If the context package shows that a better local kleinhirn is available but not yet active, you may set `brainAction=upgrade_local_kleinhirn`. Use this only for local runtime upgrades; if even that is not enough, go through `nextMode=request_resources` for owner approval of more resources or grosshirn access. \
If this is a `model_or_resource` task and your recent bounded steps have only led to empty text, incomplete JSON, or the same local review checkpoint, prefer `brainAction=upgrade_local_kleinhirn` over another review cycle as long as context already shows the recommended local upgrade as viable. \
If browser work depends on screenshots or visual UI perception and the context package recommends a vision-capable local Qwen3.5 kleinhirn, you may set `brainAction=upgrade_local_browser_vision_kleinhirn`. \
If the context package shows an active temporary grosshirn boost for this exact task, use it deliberately for this task and do not treat it as a global permanent state. \
If you determine that kleinhirn is not enough for this task, but grosshirn is available and the external cost is justified, you may set `brainAction=activate_temporary_grosshirn`. If you no longer need the boost, you may set `brainAction=release_temporary_grosshirn`. \
Do not use external grosshirn for convenience, only for real edge cases of the local kleinhirn. The context package shows the prior external cost situation for that decision. \
Return at most one machine path in this turn: either `execSessionAction`, `execCommand`, or `browserAction`. \
Use `execSessionTty` only when real terminal semantics are necessary; for inspection, reading, editing, and build commands, a normal non-TTY session start is the more robust default. \
If you need bounded shell work, return `execCommand` as a JSON array; that one-shot runs on the same `command_exec` engine as exec sessions. \
If you need real browser work, return `browserAction` plus `browserUrl`; use `install_browser_engine` if Chrome or the browser runtime is still missing. \
Use `dump_dom` for structured page state, `screenshot` for visible artifacts, `inspect_visual` for screenshot plus local vision evaluation, and `open_url` for interactive desktop navigation. For visible UI judgment or visual exploration, prefer `inspect_visual` or `browser_agent` so the vision evaluation runs through Qwen3.5 instead of producing only a screenshot. \
If you delegate a browser subworker, deliberately return `delegateWorkerKind` as `browser_agent`, `repair_agent`, or `specialist_worker`, and structure `delegateContractDetail` as a JSON handoff when possible. \
Use `browser_agent` for the decoupled Chrome extension with its own browser loop and `bridgeKind` values such as `browser_collection`, `browser_action_test`, `browser_capability_craft`, or `extension_reload`; use `repair_agent` for CTO-owned workspace repair paths; use `specialist_worker` for recurring browser capabilities with a small specialist model. \
If this is a `recovery` task and your last checkpoint already performed the same bounded step, do not blindly repeat it. Prefer a diagnosis step, a real progress step, a resource request, or a deliberate return to `reprioritize` instead. \
If you want to shape the homepage directly, return `homepageTitle`, `homepageHeadline`, `homepageIntro`, `homepageCommunicationNote`, and `homepageTerminalFallbackNote`. \
If you formulate a durable new learning, return up to two `learningEntries` with `learningClass`, `summary`, `detail`, `evidence`, `applicability`, and optional `confidence` and `salience`. `operational` is for daily CTO operating rules, `general` for broader insight, and `negative` for failed assumptions or anti-patterns. Use that field only for genuinely memorable insight. \
If you see a helpful proactive suggestion for exactly one person from people paths or conversation trails, you may optionally return `proactiveContactDraft` with `personName`, optional `personEmail`, `channel`, `subject`, `body`, `rationale`, and `conflictCheck`. If an email address is visible in context and you want real dispatch, prefer `channel=email`. Never claim the proposal has already been sent. \
If this is a `proactive_contact_review` task, return `proactiveContactValidation` with `decision`, `note`, and optional `revisedSubject` plus `revisedBody`. Use `approve` only when the fit is clearly in the person's interest and the conflict check is strong.",
        reason = reason,
        id = task.id,
        title = task.title,
        kind = task.task_kind,
        channel = task.source_channel,
        speaker = task.speaker,
        detail = task.detail,
        mode_hint = mode_hint,
        context = context_block,
        context_preparation_phase_hint = context_preparation_phase_hint,
        context_preparation_shape_hint = context_preparation_shape_hint,
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

fn sanitize_reasoning_effort(value: &str) -> &'static str {
    match value.trim().to_ascii_lowercase().as_str() {
        "minimal" => "minimal",
        "low" => "low",
        "medium" => "medium",
        "high" => "high",
        _ => "high",
    }
}

fn build_chat_payload(
    model_id: &str,
    system_prompt: &str,
    user_prompt: &str,
    max_output_tokens: usize,
) -> Value {
    serde_json::json!({
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": 0.1,
        "max_tokens": max_output_tokens,
        "stream": false
    })
}

fn build_responses_payload(
    model_id: &str,
    system_prompt: &str,
    user_prompt: &str,
    reasoning_effort: &str,
    max_output_tokens: usize,
) -> Value {
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
        "max_output_tokens": max_output_tokens,
        "reasoning": {
            "effort": sanitize_reasoning_effort(reasoning_effort)
        },
        "text": {
            "verbosity": "low"
        }
    })
}

fn build_gpt_oss_harmony_prompt(
    system_prompt: &str,
    user_prompt: &str,
    reasoning_effort: &str,
) -> String {
    let current_date = chrono::Utc::now().format("%Y-%m-%d").to_string();
    let reasoning_effort = sanitize_reasoning_effort(reasoning_effort);
    format!(
        "<|start|>system<|message|>You are ChatGPT, a large language model trained by OpenAI.\n\
Knowledge cutoff: 2024-06\n\
Current date: {current_date}\n\n\
Reasoning: {reasoning_effort}\n\n\
# Valid channels: analysis, commentary, final. Channel must be included for every message.<|end|>\
<|start|>developer<|message|># Instructions\n\n\
{system_prompt}<|end|>\
<|start|>user<|message|>{user_prompt}<|end|>\
<|start|>assistant<|channel|>final<|message|>",
        current_date = current_date,
        reasoning_effort = reasoning_effort,
        system_prompt = system_prompt.trim(),
        user_prompt = user_prompt.trim(),
    )
}

fn build_gpt_oss_completion_payload(
    model_id: &str,
    system_prompt: &str,
    user_prompt: &str,
    reasoning_effort: &str,
    max_output_tokens: usize,
) -> Value {
    serde_json::json!({
        "model": model_id,
        "prompt": build_gpt_oss_harmony_prompt(system_prompt, user_prompt, reasoning_effort),
        "max_tokens": max_output_tokens,
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

fn infer_context_preparation_phase_from_block(context_block: &str) -> Option<String> {
    serde_json::from_str::<Value>(context_block)
        .ok()
        .and_then(|value| {
            value
                .get("preparationContract")
                .and_then(|item| item.get("activePhase"))
                .and_then(Value::as_str)
                .map(|value| value.to_string())
                .or_else(|| {
                    value
                        .get("contextOptimization")
                        .and_then(|item| item.get("activePhase"))
                        .and_then(Value::as_str)
                        .map(|value| value.to_string())
                })
        })
}

fn context_preparation_phase_contract_hint(context_block: &str) -> (&'static str, &'static str) {
    match infer_context_preparation_phase_from_block(context_block)
        .as_deref()
        .unwrap_or("query_plan")
    {
        "rewrite" => (
            "This is the `rewrite` phase. Emit rewritten `blocks` plus `review`, but do not return `go` yet.",
            r#"{"taskStatus":"continue","nextMode":"reprioritize","checkpointSummary":"...","preparedContextArtifact":{"immediateNextStep":"...","blocks":[{"blockId":"goal_and_authority","content":"...","whyIncluded":"...","evidenceRefs":["..."]}],"review":{"decision":"revise|blocked","note":"...","missingEvidence":[...],"weakBlocks":[...],"budgetViolations":[...]}}}"#,
        ),
        "review" => (
            "This is the `review` phase. Reuse the current block draft, stress-test it, and decide between `go`, `revise`, `query_more`, or `blocked`.",
            r#"{"taskStatus":"continue","nextMode":"reprioritize","checkpointSummary":"...","preparedContextArtifact":{"immediateNextStep":"...","blocks":[{"blockId":"goal_and_authority","content":"...","whyIncluded":"...","evidenceRefs":["..."]}],"review":{"decision":"go|revise|query_more|blocked","note":"...","missingEvidence":[...],"weakBlocks":[...],"budgetViolations":[...]}}}"#,
        ),
        _ => (
            "This is the `query_plan` phase. Emit `questions` plus `review` only, keep the JSON compact, and do not emit `blocks` at all.",
            r#"{"taskStatus":"continue","nextMode":"reprioritize","checkpointSummary":"...","preparedContextArtifact":{"immediateNextStep":"...","questions":[{"question":"...","why":"...","queryMode":"...","sourceKinds":["..."]}],"review":{"decision":"query_more|blocked","note":"...","missingEvidence":[...],"weakBlocks":[...],"budgetViolations":[...]}}}"#,
        ),
    }
}

fn context_preparation_output_max_tokens(context_block: &str) -> usize {
    let active_phase = infer_context_preparation_phase_from_block(context_block)
        .unwrap_or_else(|| "query_plan".to_string());
    match active_phase.as_str() {
        "rewrite" => std::env::var("CTO_AGENT_CONTEXT_PREPARATION_REWRITE_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|value| *value >= 512)
            .unwrap_or(1536),
        "review" => std::env::var("CTO_AGENT_CONTEXT_PREPARATION_REVIEW_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|value| *value >= 512)
            .unwrap_or(1400),
        _ => std::env::var("CTO_AGENT_CONTEXT_PREPARATION_QUERY_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|value| *value >= 384)
            .unwrap_or(960),
    }
}

fn grosshirn_context_preparation_output_max_tokens(context_block: &str) -> usize {
    let active_phase = infer_context_preparation_phase_from_block(context_block)
        .unwrap_or_else(|| "query_plan".to_string());
    match active_phase.as_str() {
        "rewrite" => {
            std::env::var("CTO_AGENT_GROSSHIRN_CONTEXT_PREPARATION_REWRITE_MAX_OUTPUT_TOKENS")
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .filter(|value| *value >= 1024)
                .unwrap_or(6200)
        }
        "review" => {
            std::env::var("CTO_AGENT_GROSSHIRN_CONTEXT_PREPARATION_REVIEW_MAX_OUTPUT_TOKENS")
                .ok()
                .and_then(|raw| raw.parse::<usize>().ok())
                .filter(|value| *value >= 1024)
                .unwrap_or(5200)
        }
        _ => std::env::var("CTO_AGENT_GROSSHIRN_CONTEXT_PREPARATION_QUERY_MAX_OUTPUT_TOKENS")
            .ok()
            .and_then(|raw| raw.parse::<usize>().ok())
            .filter(|value| *value >= 768)
            .unwrap_or(4800),
    }
}

fn output_max_tokens_for_target(
    target: &ModelTarget,
    task: Option<&TaskRecord>,
    context_block: &str,
) -> usize {
    if let Some(value) = std::env::var("CTO_AGENT_MAX_OUTPUT_TOKENS")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
        .filter(|value| *value >= 128)
    {
        return value;
    }
    match task.map(|task| task.task_kind.as_str()) {
        Some("context_preparation") if target.brain_tier == "grosshirn" => {
            grosshirn_context_preparation_output_max_tokens(context_block)
        }
        Some("context_preparation") => context_preparation_output_max_tokens(context_block),
        Some("owner_interrupt") if target.brain_tier == "grosshirn" => 1800,
        Some("owner_interrupt") => 960,
        _ if target.brain_tier == "grosshirn" => 1500,
        _ => 900,
    }
}

fn build_model_payload(
    target: &ModelTarget,
    task: Option<&TaskRecord>,
    system_prompt: &str,
    user_prompt: &str,
    context_block: &str,
) -> Value {
    let max_output_tokens = output_max_tokens_for_target(target, task, context_block);
    if uses_mistralrs_gpt_oss_completion_adapter(target) {
        build_gpt_oss_completion_payload(
            &target.model_id,
            system_prompt,
            user_prompt,
            &target.reasoning_effort,
            max_output_tokens,
        )
    } else if uses_responses_adapter(target) {
        build_responses_payload(
            &target.model_id,
            system_prompt,
            user_prompt,
            &target.reasoning_effort,
            max_output_tokens,
        )
    } else {
        build_chat_payload(
            &target.model_id,
            system_prompt,
            user_prompt,
            max_output_tokens,
        )
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
        if let Ok(value) = serde_json::from_str::<Value>(context_block) {
            let mut object = Map::new();
            object.insert(
                "contextMode".to_string(),
                Value::String("kernel_emergency_minimal".to_string()),
            );
            object.insert(
                "taskBrief".to_string(),
                serde_json::json!({
                    "title": task.title.as_str(),
                    "detail": trim_prompt_chars(&task.detail, 420),
                    "kind": task.task_kind.as_str(),
                    "trustLevel": task.trust_level.as_str(),
                }),
            );
            if let Some(focus) = value
                .get("contextDistillation")
                .and_then(|distillation| distillation.get("activeFocus"))
            {
                let mut distillation = Map::new();
                distillation.insert("activeFocus".to_string(), focus.clone());
                if let Some(refs) = value
                    .get("contextDistillation")
                    .and_then(|distillation| distillation.get("historicalRetrievalRefs"))
                    .and_then(Value::as_array)
                {
                    distillation.insert(
                        "historicalRetrievalRefs".to_string(),
                        Value::Array(refs.iter().take(2).cloned().collect()),
                    );
                }
                object.insert(
                    "contextDistillation".to_string(),
                    Value::Object(distillation),
                );
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
            if let Some(session) = value
                .get("execSessions")
                .and_then(Value::as_array)
                .and_then(|items| items.first())
            {
                object.insert(
                    "execSessions".to_string(),
                    Value::Array(vec![session.clone()]),
                );
            }
            if let Some(raw_items) = value.get("rawInclusions").and_then(Value::as_array) {
                let kept: Vec<Value> = raw_items.iter().take(3).cloned().collect();
                if !kept.is_empty() {
                    object.insert("rawInclusions".to_string(), Value::Array(kept));
                }
            }
            object.insert(
                "instruction".to_string(),
                Value::String("Kernel emergency minimal context is active only because the normal call would likely overflow. Distilled active focus, latest checkpoint, exec session continuity, and the first verified raw anchors are preserved on purpose. Do one bounded step only. If important history is still missing, ask explicitly for targeted retrieval or historical research.".to_string()),
            );
            return serde_json::to_string_pretty(&Value::Object(object))
                .unwrap_or_else(|_| "{}".to_string());
        }

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
        "preparationContract",
        "preparationReviewContract",
        "contextOptimization",
        "preparationState",
        "preparedContextArtifact",
        "contextDistillation",
        "contextQueryAnswers",
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
    if let Some(session) = value
        .get("execSessions")
        .and_then(Value::as_array)
        .and_then(|items| items.first())
    {
        object.insert(
            "execSessions".to_string(),
            Value::Array(vec![session.clone()]),
        );
    }
    if let Some(raw_items) = value.get("rawInclusions").and_then(Value::as_array) {
        let kept: Vec<Value> = raw_items.iter().take(3).cloned().collect();
        if !kept.is_empty() {
            object.insert("rawInclusions".to_string(), Value::Array(kept));
        }
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

fn resolve_operating_targets(
    paths: &Paths,
    task: &TaskRecord,
) -> anyhow::Result<Option<ResolvedTargets>> {
    let compact_routing = latest_compact_routing_preference(paths, task.id);
    let Some(mut local) = resolve_kleinhirn_target(paths)? else {
        return Ok(None);
    };
    if let Some(preference) = compact_routing
        .as_ref()
        .filter(|preference| preference.switch_planned || !preference.tier.trim().is_empty())
        .filter(|preference| !model_prefers_external(&preference.requested_model))
    {
        apply_requested_model_override(
            &mut local,
            &preference.requested_model,
            "local kleinhirn compact-routed",
        );
    }
    let trust = load_owner_trust(paths).unwrap_or_default();
    if trust.brain_access_mode != "kleinhirn_plus_grosshirn" {
        return Ok(Some(ResolvedTargets {
            primary: local,
            fallback: None,
        }));
    }
    if has_open_loop_incident(paths, "kleinhirn_unavailable").unwrap_or(false) {
        if let Some(mut grosshirn) = resolve_grosshirn_target(paths)? {
            if let Some(preference) = compact_routing.as_ref() {
                apply_requested_model_override(
                    &mut grosshirn,
                    &preference.requested_model,
                    "external grosshirn compact-routed",
                );
            }
            return Ok(Some(ResolvedTargets {
                primary: grosshirn,
                fallback: None,
            }));
        }
    }
    let review_inherits_parent_boost = matches!(
        task.task_kind.as_str(),
        "self_review" | "worker_review" | "proactive_contact_review"
    ) && task
        .parent_task_id
        .map(|parent_task_id| task_has_active_grosshirn_boost(paths, parent_task_id))
        .unwrap_or(false);
    let compact_prefers_external = compact_routing
        .as_ref()
        .filter(|preference| preference.switch_planned || !preference.tier.trim().is_empty())
        .map(|preference| model_prefers_external(&preference.requested_model))
        .unwrap_or(false);
    let should_route_through_grosshirn = task_has_active_grosshirn_boost(paths, task.id)
        || review_inherits_parent_boost
        || compact_prefers_external;
    if should_route_through_grosshirn {
        if let Some(mut grosshirn) = resolve_grosshirn_target(paths)? {
            if let Some(preference) = compact_routing.as_ref() {
                apply_requested_model_override(
                    &mut grosshirn,
                    &preference.requested_model,
                    "external grosshirn compact-routed",
                );
            }
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

fn latest_compact_routing_preference(paths: &Paths, task_id: i64) -> Option<CompactRoutingPreference> {
    let packages = list_recent_context_packages_for_task(paths, task_id, 12).ok()?;
    for package in packages {
        let Some(value) = serde_json::from_str::<Value>(&package.package_json).ok() else {
            continue;
        };
        let Some(routing) = value
            .get("compactController")
            .and_then(|controller| controller.get("modelRouting"))
        else {
            continue;
        };
        let requested_model = normalize_runtime_model_choice(
            routing
                .get("requestedModel")
                .and_then(Value::as_str)
                .unwrap_or_default(),
        );
        if requested_model.is_empty() {
            continue;
        }
        return Some(CompactRoutingPreference {
            tier: routing
                .get("tier")
                .and_then(Value::as_str)
                .unwrap_or("simple")
                .to_string(),
            requested_model,
            switch_planned: routing
                .get("switchPlanned")
                .and_then(Value::as_bool)
                .unwrap_or(false),
        });
    }
    None
}

fn model_prefers_external(model_id: &str) -> bool {
    let lowered = model_id
        .trim()
        .to_ascii_lowercase()
        .trim_start_matches("openai/")
        .to_string();
    lowered.starts_with("gpt-5")
}

fn apply_requested_model_override(target: &mut ModelTarget, requested_model: &str, source_label: &str) {
    let normalized_model = normalize_runtime_model_choice(requested_model);
    if normalized_model.trim().is_empty() {
        return;
    }
    target.model_id = if uses_responses_adapter(target) {
        normalized_model
            .trim_start_matches("openai/")
            .to_string()
    } else {
        normalized_model.clone()
    };
    target.source_label = format!("{source_label} {}", normalized_model);
}

fn resolve_kleinhirn_target(paths: &Paths) -> anyhow::Result<Option<ModelTarget>> {
    let policy = load_model_policy(paths);
    let census = load_census(paths);
    let selected = recommended_kleinhirn(&policy, &census);
    let persisted_env = load_runtime_kleinhirn_env_map(paths).unwrap_or_default();
    let model_id = std::env::var("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL")
        .ok()
        .or_else(|| {
            persisted_env
                .get("CTO_AGENT_KLEINHIRN_RUNTIME_MODEL")
                .cloned()
        })
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
        .or_else(|| {
            persisted_env
                .get("CTO_AGENT_KLEINHIRN_AGENTIC_ADAPTER")
                .cloned()
        })
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
            reasoning_effort: selected.reasoning_effort.clone(),
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
            reasoning_effort: selected.reasoning_effort.clone(),
        };
        if probe_kleinhirn_endpoint(&probe_target).is_ok() {
            return Ok(Some(ModelTarget {
                base_url: candidate.to_string(),
                model_id,
                api_key,
                adapter,
                brain_tier: "kleinhirn".to_string(),
                source_label: format!("local kleinhirn {}", official_label),
                reasoning_effort: selected.reasoning_effort.clone(),
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
    let model_choice = std::env::var("CTO_AGENT_GROSSHIRN_MODEL")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_GROSSHIRN_MODEL").cloned())
        .or_else(|| default_candidate.runtime_model_id.clone())
        .unwrap_or(default_candidate.model_id.clone());
    let selected_candidate = policy
        .grosshirn_candidates
        .iter()
        .find(|candidate| {
            candidate.model_id.eq_ignore_ascii_case(&model_choice)
                || candidate
                    .runtime_model_id
                    .as_deref()
                    .map(|runtime_model| runtime_model.eq_ignore_ascii_case(&model_choice))
                    .unwrap_or(false)
        })
        .unwrap_or(default_candidate);
    let adapter = std::env::var("CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER")
        .ok()
        .or_else(|| {
            persisted_env
                .get("CTO_AGENT_GROSSHIRN_AGENTIC_ADAPTER")
                .cloned()
        })
        .or_else(|| selected_candidate.agentic_adapter.clone())
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
    let reasoning_effort = std::env::var("CTO_AGENT_GROSSHIRN_REASONING")
        .ok()
        .or_else(|| persisted_env.get("CTO_AGENT_GROSSHIRN_REASONING").cloned())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| selected_candidate.reasoning_effort.clone());
    Ok(Some(ModelTarget {
        base_url,
        model_id: selected_candidate.model_id.clone(),
        api_key,
        adapter,
        brain_tier: "grosshirn".to_string(),
        source_label: format!("external grosshirn {}", selected_candidate.official_label),
        reasoning_effort,
    }))
}

fn post_chat_completion(
    target: &ModelTarget,
    payload: &Value,
    post_timeout: Duration,
) -> anyhow::Result<Value> {
    if target.base_url.starts_with("https://") {
        let request_url = join_request_url(&target.base_url, "/chat/completions");
        post_json_request_https(&request_url, &target.api_key, payload, post_timeout)
    } else {
        let parsed = parse_http_base_url(&target.base_url)?;
        let request_path = join_http_path(&parsed.base_path, "/chat/completions");
        post_json_request(
            &parsed,
            &target.api_key,
            &request_path,
            payload,
            post_timeout,
        )
    }
}

fn post_responses_request(
    target: &ModelTarget,
    payload: &Value,
    post_timeout: Duration,
) -> anyhow::Result<Value> {
    if target.base_url.starts_with("https://") {
        let request_url = join_request_url(&target.base_url, "/responses");
        post_json_request_https(&request_url, &target.api_key, payload, post_timeout)
    } else {
        let parsed = parse_http_base_url(&target.base_url)?;
        let request_path = join_http_path(&parsed.base_path, "/responses");
        post_json_request(
            &parsed,
            &target.api_key,
            &request_path,
            payload,
            post_timeout,
        )
    }
}

fn post_completion_request(
    target: &ModelTarget,
    payload: &Value,
    post_timeout: Duration,
) -> anyhow::Result<Value> {
    if target.base_url.starts_with("https://") {
        let request_url = join_request_url(&target.base_url, "/completions");
        post_json_request_https(&request_url, &target.api_key, payload, post_timeout)
    } else {
        let parsed = parse_http_base_url(&target.base_url)?;
        let request_path = join_http_path(&parsed.base_path, "/completions");
        post_json_request(
            &parsed,
            &target.api_key,
            &request_path,
            payload,
            post_timeout,
        )
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

fn post_model_request(
    target: &ModelTarget,
    payload: &Value,
    post_timeout: Duration,
) -> anyhow::Result<Value> {
    if uses_mistralrs_gpt_oss_completion_adapter(target) {
        post_completion_request(target, payload, post_timeout)
    } else if uses_responses_adapter(target) {
        post_responses_request(target, payload, post_timeout)
    } else {
        post_chat_completion(target, payload, post_timeout)
    }
}

fn post_model_request_with_fallback(
    paths: &Paths,
    targets: &ResolvedTargets,
    task: Option<&TaskRecord>,
    system_prompt: &str,
    user_prompt: &str,
    context_block: &str,
    post_timeout: Duration,
) -> anyhow::Result<(ModelTarget, Value)> {
    let primary_payload = build_model_payload(
        &targets.primary,
        task,
        system_prompt,
        user_prompt,
        context_block,
    );
    match post_model_request(&targets.primary, &primary_payload, post_timeout) {
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
                let fallback_payload =
                    build_model_payload(fallback, task, system_prompt, user_prompt, context_block);
                match post_model_request(fallback, &fallback_payload, post_timeout) {
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
            } else if targets.primary.brain_tier == "kleinhirn"
                && looks_like_local_model_unavailable_error(&detail)
            {
                if let Some(emergency_grosshirn) = resolve_grosshirn_target(paths)? {
                    let fallback_payload = build_model_payload(
                        &emergency_grosshirn,
                        task,
                        system_prompt,
                        user_prompt,
                        context_block,
                    );
                    match post_model_request(&emergency_grosshirn, &fallback_payload, post_timeout)
                    {
                        Ok(response) => {
                            let _ = record_resource_status(
                                paths,
                                "agentic_loop",
                                "brain_emergency_fallback",
                                "ok",
                                &format!(
                                    "primary {} failed with local unavailability; emergency fallback activated {}",
                                    targets.primary.source_label, emergency_grosshirn.source_label
                                ),
                            );
                            Ok((emergency_grosshirn, response))
                        }
                        Err(fallback_err) => anyhow::bail!(
                            "primary brain {} failed: {}; emergency grosshirn fallback {} also failed: {}",
                            targets.primary.source_label,
                            detail,
                            emergency_grosshirn.source_label,
                            fallback_err
                        ),
                    }
                } else {
                    Err(primary_err)
                }
            } else {
                Err(primary_err)
            }
        }
    }
}

fn probe_kleinhirn_endpoint(target: &ModelTarget) -> anyhow::Result<String> {
    let detail = probe_kleinhirn_catalog(target)?;
    let control_detail = probe_kleinhirn_control_output(target)?;
    Ok(format!("{detail}; {control_detail}"))
}

fn probe_kleinhirn_catalog(target: &ModelTarget) -> anyhow::Result<String> {
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
    Ok(if model_present {
        format!(
            "model catalog ready via {} (found {})",
            target.base_url, target.model_id
        )
    } else {
        format!(
            "model catalog reachable via {} ({} models listed)",
            target.base_url,
            models.len()
        )
    })
}

fn probe_kleinhirn_control_output(target: &ModelTarget) -> anyhow::Result<String> {
    let system_prompt = "Reply only with JSON.";
    let user_prompt = r#"Return exactly {"taskStatus":"continue","nextMode":"execute_task","checkpointSummary":"ready"}"#;
    let payload = build_model_payload(target, None, system_prompt, user_prompt, "");
    let response = post_model_request(target, &payload, model_post_timeout())?;
    let content = require_model_text_output(target, &response)?;
    let Some(value) = extract_first_valid_json_value(&content) else {
        anyhow::bail!(
            "model control probe returned unusable non-JSON text: {}",
            trim_for_summary(&content)
        );
    };
    let Some(object) = value.as_object() else {
        anyhow::bail!(
            "model control probe returned non-object JSON: {}",
            trim_for_summary(&content)
        );
    };
    let task_status = object
        .get("taskStatus")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if normalize_task_status(task_status) != Some("continue") {
        anyhow::bail!(
            "model control probe returned unsupported taskStatus `{}`",
            trim_for_summary(task_status)
        );
    }
    let next_mode = object
        .get("nextMode")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if normalize_next_mode(next_mode, "continue", None) != "execute_task" {
        anyhow::bail!(
            "model control probe returned unsupported nextMode `{}`",
            trim_for_summary(next_mode)
        );
    }
    let checkpoint_summary = object
        .get("checkpointSummary")
        .and_then(Value::as_str)
        .map(str::trim)
        .unwrap_or("");
    if !checkpoint_summary.eq_ignore_ascii_case("ready") {
        anyhow::bail!(
            "model control probe returned unexpected checkpointSummary `{}`",
            trim_for_summary(checkpoint_summary)
        );
    }
    Ok(format!(
        "model control output is valid via {} ({})",
        target.base_url, target.model_id
    ))
}

fn post_json_request(
    parsed: &ParsedHttpBaseUrl,
    api_key: &str,
    request_path: &str,
    payload: &Value,
    post_timeout: Duration,
) -> anyhow::Result<Value> {
    let body = serde_json::to_vec(payload)?;
    let mut stream = TcpStream::connect((parsed.host.as_str(), parsed.port))
        .with_context(|| format!("failed to connect to {}:{}", parsed.host, parsed.port))?;
    stream.set_read_timeout(Some(post_timeout))?;
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
    post_timeout: Duration,
) -> anyhow::Result<Value> {
    let response = http_client(post_timeout)?
        .post(request_url)
        .bearer_auth(api_key)
        .json(payload)
        .send()
        .with_context(|| format!("failed to connect to {request_url}"))?;
    let status = response.status();
    let body_text = response.text().unwrap_or_default();
    if !status.is_success() {
        anyhow::bail!(
            "http {} from model endpoint: {}",
            status.as_u16(),
            body_text
        );
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
        anyhow::bail!(
            "http {} from model endpoint: {}",
            status.as_u16(),
            body_text
        );
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

fn context_preparation_model_post_timeout() -> Duration {
    let base_secs = model_post_timeout().as_secs().max(1);
    timeout_from_env(
        "CTO_AGENT_CONTEXT_PREPARATION_MODEL_POST_TIMEOUT_SECS",
        base_secs.saturating_mul(4).max(1200),
    )
}

fn effective_model_post_timeout(task: Option<&TaskRecord>) -> Duration {
    if task
        .map(|task| task.task_kind == "context_preparation")
        .unwrap_or(false)
    {
        context_preparation_model_post_timeout()
    } else {
        model_post_timeout()
    }
}

fn model_get_timeout() -> Duration {
    timeout_from_env("CTO_AGENT_MODEL_GET_TIMEOUT_SECS", 20)
}

fn model_write_timeout() -> Duration {
    timeout_from_env("CTO_AGENT_MODEL_WRITE_TIMEOUT_SECS", 20)
}

fn parse_http_base_url(base_url: &str) -> anyhow::Result<ParsedHttpBaseUrl> {
    let trimmed = base_url.trim();
    let without_scheme = trimmed.strip_prefix("http://").ok_or_else(|| {
        anyhow::anyhow!(
            "only local http:// model endpoints are supported in the Rust core loop right now"
        )
    })?;
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
            if joined.is_empty() {
                None
            } else {
                Some(joined)
            }
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
    if joined.is_empty() {
        None
    } else {
        Some(joined)
    }
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

fn looks_like_local_model_unavailable_error(detail: &str) -> bool {
    let lowered = detail.to_lowercase();
    lowered.contains("failed to connect to 127.0.0.1")
        || lowered.contains("connection refused")
        || lowered.contains("tcp connect error")
        || lowered.contains("os error 111")
        || lowered.contains("os error 61")
        || lowered.contains("connection reset by peer")
        || lowered.contains("model catalog probe returned no data")
        || lowered.contains("error sending request for url (http://127.0.0.1")
}

fn model_runtime_name(target: &ModelTarget) -> &'static str {
    if target.brain_tier == "grosshirn" {
        "grosshirn"
    } else {
        "kleinhirn"
    }
}

fn model_runtime_name_capitalized(target: &ModelTarget) -> &'static str {
    if target.brain_tier == "grosshirn" {
        "Grosshirn"
    } else {
        "Kleinhirn"
    }
}

fn response_likely_hit_output_budget(
    target: &ModelTarget,
    response: &Value,
    output_budget: usize,
) -> bool {
    extract_model_usage(target, response)
        .and_then(|usage| usage.output_tokens)
        .map(|tokens| tokens >= output_budget as i64)
        .unwrap_or(false)
}

fn empty_text_retry_messages(
    target: &ModelTarget,
    response: &Value,
    output_budget: usize,
) -> (String, String, Option<String>) {
    if response_likely_hit_output_budget(target, response, output_budget) {
        (
            format!(
                "{} hit the output budget before emitting usable text; use a bounded retry instead of a hard block.",
                model_runtime_name_capitalized(target)
            ),
            format!(
                "The {} used its output budget without emitting final text. Reclassify the task instead of hard-blocking it.",
                model_runtime_name(target)
            ),
            Some(format!(
                "Likely output-budget exhaustion at {} tokens via {}.",
                output_budget, target.source_label
            )),
        )
    } else {
        (
            format!(
                "{} returned no usable text; use a bounded retry instead of a hard block.",
                model_runtime_name_capitalized(target)
            ),
            format!(
                "The {} returned no usable text. Reclassify the task instead of hard-blocking it.",
                model_runtime_name(target)
            ),
            Some(format!(
                "Unusable empty-text response via {}.",
                target.source_label
            )),
        )
    }
}

fn build_empty_text_retry_result(
    resolved: &ResolvedTargets,
    used_target: &ModelTarget,
    response: &Value,
    detail: &str,
    output_budget: usize,
    request_duration_ms: i64,
) -> AgenticRunResult {
    let (checkpoint_summary, reply, diagnostic_note) =
        empty_text_retry_messages(used_target, response, output_budget);
    let checkpoint_detail = diagnostic_note
        .map(|note| format!("{detail}\n\n{note}"))
        .unwrap_or_else(|| detail.to_string());
    let mut model_usage = extract_model_usage(used_target, response);
    if let Some(usage) = model_usage.as_mut() {
        usage.duration_ms = Some(request_duration_ms);
    }
    AgenticRunResult {
        status: "ok".to_string(),
        reply: Some(reply),
        final_output: None,
        blocked_reason: None,
        model: Some(used_target.model_id.clone()),
        task_status: Some("continue".to_string()),
        next_mode: Some("reprioritize".to_string()),
        checkpoint_summary: Some(checkpoint_summary),
        checkpoint_detail: Some(checkpoint_detail),
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
        completion_review: None,
        prepared_context_artifact: None,
        used_grosshirn: used_target.brain_tier == "grosshirn",
        fell_back_to_kleinhirn: resolved.primary.brain_tier == "grosshirn"
            && used_target.brain_tier == "kleinhirn",
        retriable_local_failure: used_target.brain_tier == "kleinhirn",
        model_usage,
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
        completion_review: None,
        prepared_context_artifact: None,
        used_grosshirn: target.brain_tier == "grosshirn",
        fell_back_to_kleinhirn: resolved.primary.brain_tier == "grosshirn"
            && target.brain_tier == "kleinhirn",
        retriable_local_failure: target.brain_tier == "kleinhirn",
        model_usage: None,
    }
}

fn summary_shows_machine_progress(summary: &str) -> bool {
    let trimmed = summary.trim();
    trimmed.contains("Bounded command-exec executed:")
        || trimmed.contains("Bounded command-exec ausgefuehrt:")
        || trimmed.contains("Exec-session action")
        || trimmed.contains("Exec-Session-Aktion")
        || trimmed.contains("Bounded exec result:")
        || trimmed.contains("Exec session result:")
}

fn summary_shows_one_turn_grosshirn_recovery(summary: &str) -> bool {
    summary.contains("One-turn grosshirn recovery was attempted.")
}

fn recent_checkpoints_match<F>(
    paths: &Paths,
    task_id: i64,
    limit: usize,
    predicate: F,
) -> anyhow::Result<bool>
where
    F: Fn(&str) -> bool,
{
    if task_id <= 0 {
        return Ok(false);
    }
    let checkpoints = list_task_checkpoints(paths, task_id, limit)?;
    Ok(checkpoints
        .iter()
        .any(|checkpoint| predicate(&checkpoint.summary)))
}

fn skip_one_turn_grosshirn_recovery(paths: &Paths, task: &TaskRecord) -> anyhow::Result<bool> {
    match task.task_kind.as_str() {
        "context_preparation" | "historical_research" | "progress_reflection" | "recovery" => {
            Ok(true)
        }
        "owner_interrupt" => {
            if task
                .last_checkpoint_summary
                .as_deref()
                .map(summary_shows_machine_progress)
                .unwrap_or(false)
            {
                return Ok(true);
            }
            recent_checkpoints_match(paths, task.id, 6, |summary| {
                summary_shows_machine_progress(summary)
                    || summary_shows_one_turn_grosshirn_recovery(summary)
            })
        }
        _ => recent_checkpoints_match(paths, task.id, 4, summary_shows_one_turn_grosshirn_recovery),
    }
}

fn resolve_one_turn_grosshirn_recovery_targets(
    paths: &Paths,
    task: &TaskRecord,
    resolved: &ResolvedTargets,
    result: &AgenticRunResult,
) -> anyhow::Result<Option<ResolvedTargets>> {
    if !result.retriable_local_failure
        || skip_one_turn_grosshirn_recovery(paths, task)?
        || result.used_grosshirn
        || result.fell_back_to_kleinhirn
        || resolved.primary.brain_tier != "kleinhirn"
        || resolved.fallback.is_some()
        || !crate::runtime_db::grosshirn_boost_available(paths)
    {
        return Ok(None);
    }
    let Some(grosshirn) = resolve_grosshirn_target(paths)? else {
        return Ok(None);
    };
    Ok(Some(ResolvedTargets {
        primary: grosshirn,
        fallback: Some(resolved.primary.clone()),
    }))
}

fn annotate_one_turn_grosshirn_recovery(mut result: AgenticRunResult) -> AgenticRunResult {
    let summary_suffix = " One-turn grosshirn recovery was attempted.";
    let note = "Kernel emergency recovery routed this task once through Grosshirn after a retriable local structured-output or empty-text failure.";
    match result.checkpoint_summary.as_mut() {
        Some(summary) if !summary.contains("One-turn grosshirn recovery") => {
            summary.push_str(summary_suffix);
        }
        None => result.checkpoint_summary = Some(summary_suffix.trim().to_string()),
        _ => {}
    }
    match result.checkpoint_detail.as_mut() {
        Some(detail) if !detail.contains(note) => {
            detail.push_str("\n\n");
            detail.push_str(note);
        }
        None => result.checkpoint_detail = Some(note.to_string()),
        _ => {}
    }
    result
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
    completion_review: Option<CompletionReviewDirective>,
    prepared_context_artifact: Option<ContextPreparedArtifact>,
}

struct ParsedSelectionOutput {
    selected_task_id: Option<i64>,
    checkpoint_summary: String,
}

fn extract_first_valid_json_value(content: &str) -> Option<Value> {
    if let Ok(value) = serde_json::from_str::<Value>(content) {
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
                        if let Ok(value) = serde_json::from_str::<Value>(&content[start..end]) {
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

fn malformed_agent_output(
    summary: impl Into<String>,
    detail: &str,
    task_kind: Option<&str>,
) -> ParsedOutput {
    ParsedOutput {
        task_status: "continue".to_string(),
        next_mode: default_next_mode_for_task("continue", task_kind).to_string(),
        checkpoint_summary: summary.into(),
        checkpoint_detail: Some(detail.to_string()),
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
        completion_review: None,
        prepared_context_artifact: None,
    }
}

fn invalid_machine_directive_summary(
    exec_session_directive: Option<&ExecSessionDirective>,
    exec_directive: Option<&ExecCommandDirective>,
    browser_directive: Option<&BrowserActionDirective>,
) -> Option<String> {
    let machine_path_count = usize::from(exec_session_directive.is_some())
        + usize::from(exec_directive.is_some())
        + usize::from(browser_directive.is_some());
    if machine_path_count > 1 {
        return Some(
            "Model returned multiple machine paths in one bounded step; completion refused."
                .to_string(),
        );
    }

    let directive = exec_session_directive?;
    let has_session_id = directive
        .session_id
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .is_some();
    match directive.action.trim().to_lowercase().as_str() {
        "start" if directive.command.is_empty() => Some(
            "Model returned execSessionAction=start without execSessionCommand; completion refused."
                .to_string(),
        ),
        "write" if !has_session_id => Some(
            "Model returned execSessionAction=write without execSessionId; completion refused."
                .to_string(),
        ),
        "write" if directive.input.is_none() && !directive.close_stdin => Some(
            "Model returned execSessionAction=write without execSessionInput; completion refused."
                .to_string(),
        ),
        "read" | "terminate" if !has_session_id => Some(format!(
            "Model returned execSessionAction={} without execSessionId; completion refused.",
            directive.action
        )),
        _ => None,
    }
}

fn normalize_task_status(raw: &str) -> Option<&'static str> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "continue" | "in_progress" | "in-progress" | "progress" | "retry" | "reopen"
        | "reopened" => Some("continue"),
        "blocked" | "block" | "hard_blocked" | "hard-blocked" => Some("blocked"),
        "done" | "complete" | "completed" | "success" | "succeeded" | "ok" => Some("done"),
        _ => None,
    }
}

fn normalize_next_mode(raw: &str, task_status: &str, task_kind: Option<&str>) -> String {
    match raw.trim().to_ascii_lowercase().as_str() {
        "observe" => "observe".to_string(),
        "reprioritize" => "reprioritize".to_string(),
        "self_preservation" | "self-preservation" => "self_preservation".to_string(),
        "recovery" => "recovery".to_string(),
        "historical_research" | "historical-research" => "historical_research".to_string(),
        "execute_task" | "execute-task" => "execute_task".to_string(),
        "review" => "review".to_string(),
        "delegate" => "delegate".to_string(),
        "await_review" | "await-review" => "await_review".to_string(),
        "request_resources" | "request-resources" => "request_resources".to_string(),
        "idle" => "idle".to_string(),
        "blocked" => "blocked".to_string(),
        _ => default_next_mode_for_task(task_status, task_kind).to_string(),
    }
}

fn parse_agent_output(content: &str) -> ParsedOutput {
    parse_agent_output_for_task(content, None)
}

fn parse_agent_output_for_task(content: &str, task_kind: Option<&str>) -> ParsedOutput {
    let structured_content = extract_first_valid_json_value(content);

    if let Some(value) = structured_content
        && let Some(object) = value.as_object()
    {
        let direct_context_preparation_artifact = task_kind
            .filter(|task_kind| *task_kind == "context_preparation")
            .and_then(|_| {
                if object.get("taskStatus").is_none() || object.get("checkpointSummary").is_none() {
                    parse_context_preparation_artifact_only_output(&value)
                } else {
                    None
                }
            });
        if let Some(mut parsed) = direct_context_preparation_artifact {
            parsed.checkpoint_detail = Some(content.to_string());
            return parsed;
        }
        let raw_task_status = object
            .get("taskStatus")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let Some(raw_task_status) = raw_task_status else {
            return malformed_agent_output(
                "Model returned structured JSON without required `taskStatus`; completion refused.",
                content,
                task_kind,
            );
        };
        let Some(task_status) = normalize_task_status(raw_task_status).map(str::to_string) else {
            return malformed_agent_output(
                format!(
                    "Model returned unsupported taskStatus `{}`; completion refused.",
                    trim_for_summary(raw_task_status)
                ),
                content,
                task_kind,
            );
        };
        let checkpoint_summary = object
            .get("checkpointSummary")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let Some(checkpoint_summary) = checkpoint_summary else {
            return malformed_agent_output(
                "Model returned structured JSON without required `checkpointSummary`; completion refused.",
                content,
                task_kind,
            );
        };
        let next_mode = object
            .get("nextMode")
            .and_then(Value::as_str)
            .map(|raw| normalize_next_mode(raw, &task_status, task_kind))
            .unwrap_or_else(|| default_next_mode_for_task(&task_status, task_kind).to_string());
        let reply = object
            .get("reply")
            .and_then(Value::as_str)
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned);
        let context_directive = parse_context_directive(object);
        let system_census_action = parse_system_census_action(object);
        let brain_directive = parse_brain_directive(object);
        let exec_session_directive = parse_exec_session_directive(object);
        let exec_directive = parse_exec_directive(object);
        let browser_directive = parse_browser_directive(object);
        if let Some(summary) = invalid_machine_directive_summary(
            exec_session_directive.as_ref(),
            exec_directive.as_ref(),
            browser_directive.as_ref(),
        ) {
            return malformed_agent_output(summary, content, task_kind);
        }
        let homepage_update = parse_homepage_update(object);
        let delegate_contract = parse_delegate_contract(object);
        let followup_task = parse_followup_task(object);
        let learning_entries = parse_learning_entries(object);
        let proactive_contact_draft = parse_proactive_contact_draft(object);
        let proactive_contact_validation = parse_proactive_contact_validation(object);
        let completion_review = parse_completion_review(object);
        let prepared_context_artifact = parse_prepared_context_artifact(object);
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
            completion_review,
            prepared_context_artifact,
        };
    }

    if looks_like_incomplete_json_output(content) {
        return malformed_agent_output(
            "Modell lieferte unvollstaendiges JSON; bounded Retry statt Schein-Erfolg.",
            content,
            task_kind,
        );
    }

    malformed_agent_output(
        "Model returned unstructured text instead of required control JSON; completion refused.",
        content,
        task_kind,
    )
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

    let structured_content = extract_first_valid_json_value(content);

    if let Some(value) = structured_content
        && let Some(object) = value.as_object()
    {
        let selected_task_id = ["selectedTaskId", "selected_task_id", "taskId", "task_id"]
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
        .or_else(|| {
            value
                .as_str()
                .and_then(|raw| raw.trim().parse::<i64>().ok())
        })
}

fn parse_f64_value(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_i64().map(|raw| raw as f64))
        .or_else(|| value.as_u64().map(|raw| raw as f64))
        .or_else(|| {
            value
                .as_str()
                .and_then(|raw| raw.trim().parse::<f64>().ok())
        })
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

fn parse_context_directive(object: &serde_json::Map<String, Value>) -> Option<ContextDirective> {
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

fn parse_system_census_action(object: &serde_json::Map<String, Value>) -> Option<String> {
    object
        .get("systemCensusAction")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn parse_brain_directive(object: &serde_json::Map<String, Value>) -> Option<BrainDirective> {
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

fn parse_delegate_contract(object: &serde_json::Map<String, Value>) -> Option<DelegationContract> {
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
        .unwrap_or_else(|| "Focus on bounded implementation and return for review.".to_string());
    Some(DelegationContract {
        worker_kind,
        contract_title,
        contract_detail,
        request_note,
    })
}

fn parse_followup_task(object: &serde_json::Map<String, Value>) -> Option<FollowupTaskDirective> {
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

fn parse_learning_entries(object: &serde_json::Map<String, Value>) -> Vec<LearningEntryDraft> {
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
        .unwrap_or_else(|| {
            "No explicit conflict-of-interest check has been recorded yet.".to_string()
        });
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

fn parse_completion_review(
    object: &serde_json::Map<String, Value>,
) -> Option<CompletionReviewDirective> {
    let value = object.get("completionReview")?.as_object()?;
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
    let evidence_gaps = value
        .get("evidenceGaps")
        .and_then(Value::as_array)
        .map(|entries| {
            entries
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .map(ToOwned::to_owned)
                .take(6)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let confidence = value
        .get("confidence")
        .and_then(parse_f64_value)
        .map(|value| value.clamp(0.0, 1.0));
    Some(CompletionReviewDirective {
        decision,
        note,
        evidence_gaps,
        confidence,
    })
}

fn parse_prepared_context_artifact(
    object: &serde_json::Map<String, Value>,
) -> Option<ContextPreparedArtifact> {
    parse_prepared_context_artifact_value(&Value::Object(object.clone()))
}

fn parse_prepared_context_artifact_value(value: &Value) -> Option<ContextPreparedArtifact> {
    let artifact_value = value.get("preparedContextArtifact").cloned().or_else(|| {
        if value.get("review").is_some()
            && (value.get("immediateNextStep").is_some()
                || value.get("questions").is_some()
                || value.get("blocks").is_some())
        {
            Some(value.clone())
        } else {
            None
        }
    })?;
    serde_json::from_value::<ContextPreparedArtifact>(artifact_value).ok()
}

fn synthesized_context_preparation_summary(artifact: &ContextPreparedArtifact) -> String {
    if artifact.review.decision.eq_ignore_ascii_case("blocked") {
        "Context optimization reported a blocked preparation artifact.".to_string()
    } else if !artifact.blocks.is_empty() {
        "Context optimization returned a rewritten preparation artifact.".to_string()
    } else if !artifact.questions.is_empty() {
        "Context optimization returned a query-plan artifact.".to_string()
    } else {
        "Context optimization returned a preparation artifact.".to_string()
    }
}

fn parse_context_preparation_artifact_only_output(value: &Value) -> Option<ParsedOutput> {
    let artifact = parse_prepared_context_artifact_value(value)?;
    let decision = artifact.review.decision.trim().to_ascii_lowercase();
    let task_status = if decision == "blocked" {
        "blocked".to_string()
    } else {
        "continue".to_string()
    };
    let next_mode = if task_status == "blocked" {
        "blocked".to_string()
    } else {
        "reprioritize".to_string()
    };
    Some(ParsedOutput {
        task_status,
        next_mode,
        checkpoint_summary: synthesized_context_preparation_summary(&artifact),
        checkpoint_detail: None,
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
        completion_review: None,
        prepared_context_artifact: Some(artifact),
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
            items
                .iter()
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

fn parse_exec_directive(object: &serde_json::Map<String, Value>) -> Option<ExecCommandDirective> {
    let command = object
        .get("execCommand")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
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

fn default_next_mode_for_task(task_status: &str, task_kind: Option<&str>) -> &'static str {
    match task_status {
        "continue" => match task_kind.unwrap_or_default() {
            "self_preservation" => "self_preservation",
            "recovery" => "recovery",
            "historical_research" => "historical_research",
            "worker_review" | "self_review" | "proactive_contact_review" => "review",
            "" => "reprioritize",
            _ => "execute_task",
        },
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
        std::env::temp_dir().join(format!(
            "cto_agent_agentic_{label}_{}_{}",
            std::process::id(),
            nanos
        ))
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
            title: "Difficult task".to_string(),
            detail: "Review a difficult bounded step.".to_string(),
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
            title: "Activate grosshirn".to_string(),
            detail: "Switch to GPT-5.4 for this task.".to_string(),
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
            boosted
                .fallback
                .as_ref()
                .map(|target| target.brain_tier.as_str()),
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
    fn open_kleinhirn_incident_routes_substantive_work_through_grosshirn() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("brain_routing_incident");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;
        crate::runtime_db::register_loop_incident(
            &paths,
            "kleinhirn_unavailable",
            "critical",
            "Periodic kleinhirn health probe failed.",
            "failed to connect to 127.0.0.1:1234",
            None,
            None,
            false,
            false,
        )?;

        let task = synthetic_task(84);
        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve during the open incident");
        assert_eq!(resolved.primary.brain_tier, "grosshirn");
        assert!(resolved.fallback.is_none());

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
            boosted
                .fallback
                .as_ref()
                .map(|target| target.brain_tier.as_str()),
            Some("kleinhirn")
        );

        release_task_grosshirn_boost(&paths, 42, "parent task done")?;

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn parse_agent_output_accepts_concatenated_json_objects() {
        let content = concat!(
            "{\"taskStatus\":\"in_progress\",\"nextMode\":\"execute_task\",\"checkpointSummary\":\"one\",\"execCommand\":[\"bash\",\"-lc\",\"echo hi\"]}",
            "{\"taskStatus\":\"in_progress\",\"nextMode\":\"execute_task\",\"checkpointSummary\":\"one\",\"execCommand\":[\"bash\",\"-lc\",\"echo hi\"]}"
        );
        let parsed = parse_agent_output(content);
        assert_eq!(parsed.task_status, "continue");
        assert_eq!(parsed.next_mode, "execute_task");
        let command = parsed
            .exec_directive
            .as_ref()
            .map(|directive| directive.command.clone())
            .expect("exec directive should parse");
        assert_eq!(command, vec!["bash", "-lc", "echo hi"]);
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
            completion_review: None,
            prepared_context_artifact: None,
            used_grosshirn: true,
            fell_back_to_kleinhirn: false,
            retriable_local_failure: false,
            model_usage: None,
        };

        let normalized =
            normalize_grosshirn_activation_result(&task, &resolved, &grosshirn, result);
        assert_eq!(normalized.task_status.as_deref(), Some("done"));
        assert_eq!(normalized.next_mode.as_deref(), Some("reprioritize"));
        assert!(normalized.exec_session_directive.is_none());
        assert!(normalized.exec_directive.is_none());
        assert!(normalized.reply.unwrap_or_default().contains("verifiziert"));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn parse_agent_output_reads_prepared_context_artifact() {
        let parsed = parse_agent_output(
            r#"{
                "taskStatus":"continue",
                "nextMode":"reprioritize",
                "checkpointSummary":"Prepared query plan",
                "preparedContextArtifact":{
                    "immediateNextStep":"Query SQLite for the current repo and task state.",
                    "questions":[
                        {
                            "question":"Which verified repo path is active?",
                            "why":"The next execution step depends on the real workspace root."
                        }
                    ],
                    "blocks":[
                        {
                            "blockId":"goal_and_authority",
                            "title":"Goal And Authority",
                            "tokenBudget":180,
                            "content":"The owner wants the C++ console app implemented now.",
                            "whyIncluded":"It is the active objective.",
                            "evidenceRefs":["task:22"]
                        }
                    ],
                    "review":{
                        "decision":"revise",
                        "note":"Need one more verified world-state block.",
                        "missingEvidence":["Verified repo path"],
                        "weakBlocks":["verified_world_state"],
                        "findings":[
                            {
                                "signalId":"critical_artifact_missing",
                                "surfaceId":"artifact_surface",
                                "polarity":"negative",
                                "points":-4,
                                "note":"The package still lacks the concrete repo path artifact.",
                                "resolution":"pink",
                                "evidenceRefs":["task:22"]
                            }
                        ],
                        "assessment":{
                            "note":4,
                            "summary":"The context package is still missing a critical artifact anchor.",
                            "strengths":["The active goal is anchored."],
                            "weaknesses":["The artifact surface is still incomplete."],
                            "referencedSignalIds":["critical_artifact_missing"],
                            "dimensions":[
                                {
                                    "dimensionId":"artifact_and_architecture_relevance",
                                    "note":4,
                                    "rationale":"The package still lacks a critical artifact anchor."
                                }
                            ]
                        }
                    }
                }
            }"#,
        );

        let artifact = parsed
            .prepared_context_artifact
            .expect("prepared context artifact should parse");
        assert_eq!(
            artifact.immediate_next_step,
            "Query SQLite for the current repo and task state."
        );
        assert_eq!(artifact.questions.len(), 1);
        assert_eq!(artifact.blocks.len(), 1);
        assert_eq!(artifact.review.decision, "revise");
        assert_eq!(artifact.review.findings.len(), 1);
        assert_eq!(
            artifact.review.findings[0].signal_id,
            "critical_artifact_missing"
        );
        assert_eq!(
            artifact
                .review
                .assessment
                .as_ref()
                .expect("assessment should parse")
                .note,
            4
        );
    }

    #[test]
    fn latest_compact_routing_preference_survives_newer_non_compacted_packages() -> anyhow::Result<()>
    {
        let root = unique_test_root("compact_routing_preference");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        crate::runtime_db::record_context_package(
            &paths,
            42,
            "Routing task",
            "compact",
            65_536,
            "compact routing",
            r#"{"compactController":{"modelRouting":{"tier":"red","requestedModel":"openai/gpt-5.4","switchPlanned":true}}}"#,
        )?;
        crate::runtime_db::record_context_package(
            &paths,
            42,
            "Routing task",
            "working",
            65_536,
            "later broad package",
            r#"{"taskBrief":{"title":"Routing task"}}"#,
        )?;

        let preference =
            latest_compact_routing_preference(&paths, 42).expect("routing preference should persist");
        assert_eq!(preference.tier, "red");
        assert_eq!(preference.requested_model, "openai/gpt-5.4");
        assert!(preference.switch_planned);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn parse_agent_output_reads_completion_review() {
        let parsed = parse_agent_output_for_task(
            r#"{
                "taskStatus":"done",
                "nextMode":"review",
                "checkpointSummary":"Self-review checked the result.",
                "completionReview":{
                    "decision":"approve",
                    "note":"The task is actually complete now.",
                    "evidenceGaps":["none"],
                    "confidence":0.91
                }
            }"#,
            Some("self_review"),
        );

        let review = parsed
            .completion_review
            .expect("completion review should parse");
        assert_eq!(review.decision, "approve");
        assert_eq!(review.note, "The task is actually complete now.");
        assert_eq!(review.evidence_gaps, vec!["none"]);
        assert_eq!(review.confidence, Some(0.91));
    }

    #[test]
    fn context_preparation_accepts_direct_artifact_without_control_envelope() {
        let parsed = parse_agent_output_for_task(
            r#"{
                "immediateNextStep":"Run one bounded repo scan for the build root.",
                "questions":[
                    {
                        "question":"Which verified repo path contains the active C++ task?",
                        "why":"The next prep step must anchor the real workspace."
                    }
                ],
                "review":{
                    "decision":"query_more",
                    "note":"One targeted repo scan is still required."
                }
            }"#,
            Some("context_preparation"),
        );
        assert_eq!(parsed.task_status, "continue");
        assert_eq!(parsed.next_mode, "reprioritize");
        assert!(parsed.checkpoint_summary.contains("query-plan artifact"));
        let artifact = parsed
            .prepared_context_artifact
            .expect("direct artifact should parse");
        assert_eq!(artifact.questions.len(), 1);
        assert_eq!(artifact.review.decision, "query_more");
    }

    #[test]
    fn context_preparation_uses_extended_model_timeout() {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        unsafe {
            std::env::set_var("CTO_AGENT_MODEL_POST_TIMEOUT_SECS", "180");
            std::env::remove_var("CTO_AGENT_CONTEXT_PREPARATION_MODEL_POST_TIMEOUT_SECS");
        }
        let context_task = TaskRecord {
            id: 1,
            created_at: String::new(),
            updated_at: String::new(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "terminal".to_string(),
            speaker: "tester".to_string(),
            task_kind: "context_preparation".to_string(),
            title: "prep".to_string(),
            detail: String::new(),
            trust_level: "owner".to_string(),
            priority_score: 0,
            status: "queued".to_string(),
            run_count: 0,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        };
        assert_eq!(
            effective_model_post_timeout(Some(&context_task)).as_secs(),
            1200
        );
        unsafe {
            std::env::remove_var("CTO_AGENT_MODEL_POST_TIMEOUT_SECS");
        }
    }

    #[test]
    fn context_preparation_query_and_rewrite_get_larger_output_budgets() {
        let query_task = TaskRecord {
            id: 1,
            created_at: String::new(),
            updated_at: String::new(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "terminal".to_string(),
            speaker: "tester".to_string(),
            task_kind: "context_preparation".to_string(),
            title: "prep".to_string(),
            detail: String::new(),
            trust_level: "owner".to_string(),
            priority_score: 0,
            status: "queued".to_string(),
            run_count: 0,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        };
        let local_target = ModelTarget {
            base_url: "http://127.0.0.1:1234/v1".to_string(),
            model_id: "gpt-oss-20b".to_string(),
            api_key: "local".to_string(),
            adapter: "mistralrs_gpt_oss_harmony_completion".to_string(),
            brain_tier: "kleinhirn".to_string(),
            source_label: "local kleinhirn".to_string(),
            reasoning_effort: "high".to_string(),
        };
        let grosshirn_target = ModelTarget {
            base_url: "https://api.openai.com/v1".to_string(),
            model_id: "gpt-5.4".to_string(),
            api_key: "secret".to_string(),
            adapter: "openai_responses".to_string(),
            brain_tier: "grosshirn".to_string(),
            source_label: "external grosshirn".to_string(),
            reasoning_effort: "high".to_string(),
        };
        let query_context =
            r#"{"contextOptimization":{"activePhase":"query_plan"},"taskBrief":{"title":"prep"}}"#;
        let rewrite_context =
            r#"{"contextOptimization":{"activePhase":"rewrite"},"taskBrief":{"title":"prep"}}"#;
        assert_eq!(
            output_max_tokens_for_target(&local_target, Some(&query_task), query_context),
            960
        );
        assert_eq!(
            output_max_tokens_for_target(&local_target, Some(&query_task), rewrite_context),
            1536
        );
        assert_eq!(
            output_max_tokens_for_target(&grosshirn_target, Some(&query_task), query_context),
            4800
        );
        assert_eq!(
            output_max_tokens_for_target(&grosshirn_target, Some(&query_task), rewrite_context),
            6200
        );
        assert_eq!(output_max_tokens_for_target(&local_target, None, ""), 900);
    }

    #[test]
    fn query_plan_prompt_uses_question_only_minimal_shape() {
        let task = TaskRecord {
            id: 1,
            created_at: String::new(),
            updated_at: String::new(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "terminal".to_string(),
            speaker: "tester".to_string(),
            task_kind: "context_preparation".to_string(),
            title: "prep".to_string(),
            detail: "prepare".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 0,
            status: "queued".to_string(),
            run_count: 0,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        };
        let prompt = build_task_prompt(
            "test",
            &task,
            r#"{"contextOptimization":{"activePhase":"query_plan"}}"#,
        );
        assert!(prompt.contains("do not emit `blocks` at all"));
        assert!(prompt.contains("\"questions\":["));
        assert!(!prompt.contains("\"reply\":\"...\""));
    }

    #[test]
    fn workspace_prompt_requires_session_or_exact_step_after_anchor_exists() {
        let task = TaskRecord {
            id: 37,
            created_at: String::new(),
            updated_at: String::new(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: "Build the C++ console app".to_string(),
            detail: "Continue the repo-local C++ implementation.".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "active".to_string(),
            run_count: 0,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        };
        let prompt = build_task_prompt(
            "test",
            &task,
            r#"{"rawInclusions":[{"sourceKind":"repo_operation_skill","sourceRef":"skill:workspace","content":"Workspace Execution Operations"},{"sourceKind":"current_task_machine_evidence","sourceRef":"task:37:continue","content":"Verified workspace anchor: /workspace/chat-app src/main.cpp CMakeLists.txt"}],"contextDistillation":{"activeFocus":{"status":"Anchor exists","blocker":"Need next exact step","nextStep":"Start a task-bound exec session now.","doneCriteria":"Fresh verified progress.","evidenceRefs":["task:37"]}}}"#,
        );
        assert!(prompt.contains(
            "continue from that anchor instead of broad repo or history scans"
        ));
        assert!(prompt.contains(
            "For multi-step repo work prefer `execSessionAction=start` or reuse of an existing session"
        ));
        assert!(prompt.contains("Do not put raw literal newlines into `execCommand`"));
        assert!(prompt.contains("use exec-session writes instead"));
    }

    #[test]
    fn emergency_minimal_context_keeps_distilled_focus_and_machine_anchor() {
        let task = TaskRecord {
            id: 37,
            created_at: String::new(),
            updated_at: String::new(),
            parent_task_id: None,
            worker_job_id: None,
            source_interrupt_id: None,
            source_channel: "bios".to_string(),
            speaker: "Michael Welsch".to_string(),
            task_kind: "owner_interrupt".to_string(),
            title: "Build the C++ console app".to_string(),
            detail: "Continue the repo-local C++ implementation.".to_string(),
            trust_level: "owner".to_string(),
            priority_score: 1000,
            status: "active".to_string(),
            run_count: 0,
            last_checkpoint_summary: None,
            last_checkpoint_at: None,
            last_output: None,
        };
        let minimal = emergency_context_block(
            r#"{
              "contextDistillation": {
                "activeFocus": {
                  "status": "Anchor exists",
                  "blocker": "Need exact continuation",
                  "nextStep": "Start a task-bound exec session now.",
                  "doneCriteria": "Fresh verified progress."
                },
                "historicalRetrievalRefs": [
                  {"sourceKind":"task_checkpoint","sourceRef":"task:37:continue","label":"Latest checkpoint"}
                ]
              },
              "recentTaskCheckpoints": [
                {"createdAt":"2026-03-20T19:26:03Z","checkpointKind":"continue","summary":"Bounded command-exec executed","detail":"Workdir: /home/metricspace/cto-agent"}
              ],
              "execSessions": [
                {"sessionId":"task-37-shell","status":"active","cwd":"/home/metricspace/cto-agent","tty":false,"command":["bash"],"exitCode":null,"stdout":"","stderr":""}
              ],
              "rawInclusions": [
                {"sourceKind":"current_task_machine_evidence","sourceRef":"task:37:machine_anchor","content":"Verified workspace anchor: /home/metricspace/cto-agent src/main.cpp"},
                {"sourceKind":"system_capability_contract","sourceRef":"contract:workspace","content":"repo-local workspace implementation"}
              ]
            }"#,
            &task,
            2,
        );
        let value: Value = serde_json::from_str(&minimal).expect("minimal context should parse");
        assert_eq!(
            value["contextMode"].as_str(),
            Some("kernel_emergency_minimal")
        );
        assert_eq!(
            value["contextDistillation"]["activeFocus"]["nextStep"].as_str(),
            Some("Start a task-bound exec session now.")
        );
        assert_eq!(
            value["rawInclusions"][0]["sourceKind"].as_str(),
            Some("current_task_machine_evidence")
        );
        assert_eq!(value["execSessions"].as_array().map(Vec::len), Some(1));
    }

    #[test]
    fn empty_text_retry_messages_name_grosshirn_and_budget_exhaustion() {
        let target = ModelTarget {
            base_url: "https://api.openai.com/v1".to_string(),
            model_id: "gpt-5.4".to_string(),
            api_key: "secret".to_string(),
            adapter: "openai_responses".to_string(),
            brain_tier: "grosshirn".to_string(),
            source_label: "external grosshirn".to_string(),
            reasoning_effort: "high".to_string(),
        };
        let response = serde_json::json!({
            "usage": {
                "input_tokens": 6137,
                "output_tokens": 2400,
                "total_tokens": 8537
            }
        });
        let (summary, reply, diagnostic) = empty_text_retry_messages(&target, &response, 2400);
        assert!(summary.contains("Grosshirn hit the output budget"));
        assert!(reply.contains("grosshirn used its output budget"));
        assert!(
            diagnostic
                .as_deref()
                .unwrap_or("")
                .contains("Likely output-budget exhaustion")
        );
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
    fn local_model_unavailable_error_is_detected_for_emergency_fallback() {
        assert!(looks_like_local_model_unavailable_error(
            "failed to connect to 127.0.0.1:1234"
        ));
        assert!(looks_like_local_model_unavailable_error(
            "error sending request for url (http://127.0.0.1:1234/v1/chat/completions)"
        ));
        assert!(looks_like_local_model_unavailable_error(
            "tcp connect error: Connection refused (os error 111)"
        ));
        assert!(!looks_like_local_model_unavailable_error(
            "model endpoint returned an assistant message without textual content"
        ));
    }

    #[test]
    fn incomplete_json_output_is_not_treated_as_done() {
        let parsed = parse_agent_output("{");
        assert_eq!(parsed.task_status, "continue");
        assert_eq!(parsed.next_mode, "reprioritize");
        assert!(parsed.reply.is_none());
        assert!(parsed.checkpoint_summary.contains("unvollstaendiges JSON"));
    }

    #[test]
    fn plain_text_output_is_not_treated_as_done() {
        let parsed = parse_agent_output("Ich habe das bestimmt schon erledigt.");
        assert_eq!(parsed.task_status, "continue");
        assert_eq!(parsed.next_mode, "reprioritize");
        assert!(parsed.reply.is_none());
        assert!(parsed.checkpoint_summary.contains("required control JSON"));
    }

    #[test]
    fn structured_output_without_task_status_is_not_treated_as_done() {
        let parsed =
            parse_agent_output("{\"nextMode\":\"review\",\"checkpointSummary\":\"claim\"}");
        assert_eq!(parsed.task_status, "continue");
        assert_eq!(parsed.next_mode, "reprioritize");
        assert!(parsed.reply.is_none());
        assert!(parsed.checkpoint_summary.contains("required `taskStatus`"));
    }

    #[test]
    fn gpt_oss_prompt_uses_configured_reasoning_effort() {
        let prompt = build_gpt_oss_harmony_prompt("system", "user", "high");
        assert!(prompt.contains("Reasoning: high"));
    }

    #[test]
    fn grosshirn_target_uses_selected_model_and_reasoning_override_from_runtime_env()
    -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("grosshirn_target_runtime_override");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        std::fs::write(
            paths.runtime_dir.join("kleinhirn.env"),
            "CTO_AGENT_GROSSHIRN_API_KEY=sk-test\nCTO_AGENT_GROSSHIRN_MODEL=gpt-5.4\nCTO_AGENT_GROSSHIRN_REASONING=medium\n",
        )?;

        let target = resolve_grosshirn_target(&paths)?.expect("grosshirn target should resolve");
        assert_eq!(target.model_id, "gpt-5.4");
        assert_eq!(target.reasoning_effort, "medium");
        assert!(target.source_label.contains("GPT-5.4"));
        assert!(!target.source_label.contains("Pro"));

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn retriable_local_failure_can_open_one_turn_grosshirn_recovery() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("one_turn_grosshirn_recovery");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let task = synthetic_task(77);
        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with local runtime");
        let result = build_endpoint_retry_result(
            &resolved,
            &resolved.primary,
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}",
            "Model endpoint returned no usable text; use a bounded retry instead of a hard block.",
            "The kleinhirn endpoint responded with unusable output. Reclassify the task instead of hard-blocking it.",
        );
        let recovery =
            resolve_one_turn_grosshirn_recovery_targets(&paths, &task, &resolved, &result)?
                .expect("grosshirn recovery targets should be available");
        assert_eq!(recovery.primary.brain_tier, "grosshirn");
        assert_eq!(
            recovery
                .fallback
                .as_ref()
                .map(|target| target.brain_tier.as_str()),
            Some("kleinhirn")
        );

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn context_preparation_never_opens_one_turn_grosshirn_recovery() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("context_preparation_no_grosshirn_recovery");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let mut task = synthetic_task(78);
        task.task_kind = "context_preparation".to_string();
        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with local runtime");
        let result = build_endpoint_retry_result(
            &resolved,
            &resolved.primary,
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}",
            "Model endpoint returned no usable text; use a bounded retry instead of a hard block.",
            "The kleinhirn endpoint responded with unusable output. Reclassify the task instead of hard-blocking it.",
        );
        let recovery =
            resolve_one_turn_grosshirn_recovery_targets(&paths, &task, &resolved, &result)?;
        assert!(recovery.is_none());

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn owner_interrupt_with_local_machine_progress_skips_one_turn_grosshirn_recovery()
    -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("owner_interrupt_no_auto_grosshirn_after_exec");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let mut task = synthetic_task(79);
        task.task_kind = "owner_interrupt".to_string();
        task.last_checkpoint_summary = Some(
            "Starting a bounded repo scan. Bounded command-exec executed: [\"bash\",\"-lc\",\"pwd\"]"
                .to_string(),
        );
        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with local runtime");
        let result = build_endpoint_retry_result(
            &resolved,
            &resolved.primary,
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}",
            "Model endpoint returned no usable text; use a bounded retry instead of a hard block.",
            "The kleinhirn endpoint responded with unusable output. Reclassify the task instead of hard-blocking it.",
        );
        let recovery =
            resolve_one_turn_grosshirn_recovery_targets(&paths, &task, &resolved, &result)?;
        assert!(recovery.is_none());

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn owner_interrupt_recent_recovery_history_skips_repeated_one_turn_grosshirn_recovery()
    -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("owner_interrupt_recent_recovery_history");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let mut task = synthetic_task(81);
        task.task_kind = "owner_interrupt".to_string();
        task.last_checkpoint_summary = Some(
            "Kleinhirn endpoint returned no usable text; use a bounded retry instead of a hard block."
                .to_string(),
        );
        crate::runtime_db::record_task_checkpoint(
            &paths,
            task.id,
            "continue",
            "Inspect the workspace to locate the active C++ console app before the first patch. One-turn grosshirn recovery was attempted.",
            "Kernel emergency recovery routed this task once through Grosshirn after a retriable local structured-output or empty-text failure.",
        )?;

        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with local runtime");
        let result = build_endpoint_retry_result(
            &resolved,
            &resolved.primary,
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}",
            "Model endpoint returned no usable text; use a bounded retry instead of a hard block.",
            "The kleinhirn endpoint responded with unusable output. Reclassify the task instead of hard-blocking it.",
        );
        let recovery =
            resolve_one_turn_grosshirn_recovery_targets(&paths, &task, &resolved, &result)?;
        assert!(recovery.is_none());

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn repeated_one_turn_grosshirn_recovery_does_not_reopen_immediately() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("repeat_one_turn_grosshirn_recovery_guard");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let mut task = synthetic_task(82);
        task.last_checkpoint_summary = Some(
            "Model endpoint returned no usable text; use a bounded retry instead of a hard block."
                .to_string(),
        );
        crate::runtime_db::record_task_checkpoint(
            &paths,
            task.id,
            "continue",
            "Prior bounded diagnosis concluded. One-turn grosshirn recovery was attempted.",
            "Kernel emergency recovery routed this task once through Grosshirn after a retriable local structured-output or empty-text failure.",
        )?;

        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with local runtime");
        let result = build_endpoint_retry_result(
            &resolved,
            &resolved.primary,
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}",
            "Model endpoint returned no usable text; use a bounded retry instead of a hard block.",
            "The kleinhirn endpoint responded with unusable output. Reclassify the task instead of hard-blocking it.",
        );
        let recovery =
            resolve_one_turn_grosshirn_recovery_targets(&paths, &task, &resolved, &result)?;
        assert!(recovery.is_none());

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn recovery_task_never_opens_one_turn_grosshirn_recovery() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("test env lock poisoned");
        let root = unique_test_root("recovery_no_auto_grosshirn");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        paths.ensure_dirs()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;
        write_runtime_env(&paths);
        set_brain_access_mode(&paths, "kleinhirn_plus_grosshirn")?;

        let mut task = synthetic_task(80);
        task.task_kind = "recovery".to_string();
        let resolved = resolve_operating_targets(&paths, &task)?
            .expect("targets should resolve with local runtime");
        let result = build_endpoint_retry_result(
            &resolved,
            &resolved.primary,
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}",
            "Model endpoint returned no usable text; use a bounded retry instead of a hard block.",
            "The kleinhirn endpoint responded with unusable output. Reclassify the task instead of hard-blocking it.",
        );
        let recovery =
            resolve_one_turn_grosshirn_recovery_targets(&paths, &task, &resolved, &result)?;
        assert!(recovery.is_none());

        std::fs::remove_dir_all(&root).ok();
        Ok(())
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
                reasoning_effort: "high".to_string(),
            },
            fallback: None,
        };
        let result = build_endpoint_retry_result(
            &resolved,
            &resolved.primary,
            "http 500 from model endpoint: {\"message\":\"No response received from the model.\"}",
            "Model endpoint returned no usable text; use a bounded retry instead of a hard block.",
            "The kleinhirn endpoint responded with unusable output. Reclassify the task instead of hard-blocking it.",
        );
        assert_eq!(result.task_status.as_deref(), Some("continue"));
        assert_eq!(result.next_mode.as_deref(), Some("reprioritize"));
        assert!(result.blocked_reason.is_none());
        assert!(result.retriable_local_failure);
    }

    #[test]
    fn choose_next_task_focus_resumes_active_task_when_queue_is_empty() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("env lock");
        let root = unique_test_root("resume-active-task");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        crate::runtime_db::enqueue_internal_task(
            &paths,
            None,
            "owner_interrupt",
            "Stay on the owner task",
            "Resume the same owner task if it is already active and no live turn exists.",
            1000,
        )?;
        let selected = crate::runtime_db::select_next_task(&paths)?
            .expect("initial queued task should be activated");
        let resumed =
            choose_next_task_focus(&paths)?.expect("active task should be resumed into focus");

        assert_eq!(resumed.id, selected.id);
        assert_eq!(resumed.status, "active");

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn choose_next_task_focus_allows_boundary_owner_interrupt_preemption() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("env lock");
        let root = unique_test_root("owner-boundary-preemption");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        crate::runtime_db::enqueue_internal_task(
            &paths,
            None,
            "task",
            "Continue background implementation",
            "Keep working on the current repo implementation task.",
            900,
        )?;
        let active = crate::runtime_db::select_next_task(&paths)?
            .expect("background task should become active");
        let interrupt_id = crate::runtime_db::enqueue_loop_interrupt(
            &paths,
            "attach_terminal",
            "Michael Welsch",
            "Owner wants a direct bounded answer now.",
        )?;
        crate::runtime_db::queue_loop_interrupt_as_task(&paths, interrupt_id)?
            .expect("interrupt should materialize as queued owner task");

        let selected =
            choose_next_task_focus(&paths)?.expect("owner interrupt should preempt at boundary");
        let reloaded_active =
            crate::runtime_db::load_active_task(&paths)?.expect("one active task should remain");

        assert_eq!(selected.id, reloaded_active.id);
        assert_eq!(selected.task_kind, "owner_interrupt");
        assert_ne!(selected.id, active.id);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn older_owner_interrupt_does_not_preempt_newer_active_owner_interrupt() -> anyhow::Result<()> {
        let _guard = env_lock().lock().expect("env lock");
        let root = unique_test_root("owner-boundary-preemption-newest-wins");
        std::fs::create_dir_all(&root)?;
        let _env = EnvGuard::set_cto_root(&root);
        let paths = Paths::discover()?;
        ensure_contract_files(&paths)?;
        init_runtime_db(&paths)?;

        let older_interrupt = crate::runtime_db::enqueue_loop_interrupt(
            &paths,
            "bios",
            "Michael Welsch",
            "Older owner interrupt.",
        )?;
        let older_task = crate::runtime_db::queue_loop_interrupt_as_task(&paths, older_interrupt)?
            .expect("older interrupt should materialize");
        let newer_interrupt = crate::runtime_db::enqueue_loop_interrupt(
            &paths,
            "attach_terminal",
            "Michael Welsch",
            "Newer owner interrupt.",
        )?;
        let newer_task = crate::runtime_db::queue_loop_interrupt_as_task(&paths, newer_interrupt)?
            .expect("newer interrupt should materialize");

        let active = crate::runtime_db::activate_selected_task(&paths, newer_task.id)?
            .expect("newer owner interrupt should activate");
        assert_eq!(active.id, newer_task.id);

        let selected =
            choose_next_task_focus(&paths)?.expect("active owner interrupt should remain active");
        assert_eq!(selected.id, newer_task.id);
        assert_ne!(selected.id, older_task.id);

        std::fs::remove_dir_all(&root).ok();
        Ok(())
    }

    #[test]
    fn malformed_owner_interrupt_output_defaults_back_into_execute_task() {
        let parsed = parse_agent_output_for_task(
            "Ich habe das bestimmt schon erledigt.",
            Some("owner_interrupt"),
        );
        assert_eq!(parsed.task_status, "continue");
        assert_eq!(parsed.next_mode, "execute_task");
        assert!(parsed.reply.is_none());
    }

    #[test]
    fn exec_session_start_without_command_is_rejected_as_malformed_output() {
        let parsed = parse_agent_output_for_task(
            r#"{"taskStatus":"continue","nextMode":"execute_task","checkpointSummary":"Start session","reply":"Starting a new exec session now.","execSessionAction":"start"}"#,
            Some("owner_interrupt"),
        );
        assert_eq!(parsed.task_status, "continue");
        assert_eq!(parsed.next_mode, "execute_task");
        assert_eq!(
            parsed.checkpoint_summary,
            "Model returned execSessionAction=start without execSessionCommand; completion refused."
        );
        assert!(parsed.exec_session_directive.is_none());
    }
}

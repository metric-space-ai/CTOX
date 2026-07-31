// Ticket cases: dry runs, approvals, close/create/state core transitions,
// clarification plumbing, executability guards and failed writebacks.

use super::case_state::TicketCaseState;
use super::{
    action_rationale, canonical_approval_status, canonical_autonomy_level,
    canonical_learning_candidate_status, canonical_verification_status, collapse_inline,
    create_ticket_knowledge_load, default_execution_actions, enforce_core_transition,
    initial_case_state_for_approval_mode, load_ticket, map_audit_row, map_case_row,
    map_learning_candidate_row, map_ticket_clarification_row, mark_remote_events_outbound,
    now_iso_string, open_ticket_db, parse_domain_csv, parse_json_column, parse_json_or_empty,
    record_audit, record_ticket_sync_failure, required_evidence_for_bundle, resolve_ticket_control,
    stable_digest, sync_ticket_system, ControlBundleView, CoreEntityType, CoreEvent,
    CoreEvidenceRefs, CoreState, CoreTransitionRequest, DryRunRecordView,
    EffectiveControlResolution, LearningCandidateView, RuntimeLane, TicketAuditRecord,
    TicketCaseView, TicketClarificationRequestInput, TicketClarificationRequestView,
    TicketItemView, TicketKnowledgeLoadView, TicketLabelAssignmentView,
};
use crate::mission::ticket_adapters;
use crate::mission::ticket_protocol;
use anyhow::{anyhow, bail, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;

pub(super) fn create_dry_run(
    root: &Path,
    ticket_key: &str,
    understanding: Option<&str>,
    risk_level_override: Option<&str>,
) -> Result<DryRunRecordView> {
    let mut conn = open_ticket_db(root)?;
    let ticket = load_ticket(root, ticket_key)?.context("ticket not found")?;
    let knowledge_load = create_ticket_knowledge_load(root, ticket_key, None)?;
    if !knowledge_load.gap_domains.is_empty() {
        anyhow::bail!(
            "ticket knowledge gate: missing required knowledge domains for {}: {}",
            ticket_key,
            knowledge_load.gap_domains.join(", ")
        );
    }
    let (label_assignment, bundle, effective_control) = resolve_ticket_control(root, ticket_key)?;
    let now = now_iso_string();
    let case_id = format!("case:{}:{}", ticket_key, stable_digest(&now));
    let state = TicketCaseState::parse(initial_case_state_for_approval_mode(
        &effective_control.approval_mode,
    ))?;
    let risk_level = risk_level_override
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(bundle.default_risk_level.as_str())
        .to_string();
    enforce_ticket_case_create_transition(
        &conn,
        &case_id,
        ticket_key,
        state.as_str(),
        &label_assignment.label,
        &bundle.support_mode,
        "ctox-ticket",
        "create_dry_run",
    )?;
    conn.execute(
        r#"
        INSERT INTO ticket_cases (
            case_id, ticket_key, label, bundle_label, bundle_version, state, approval_mode,
            autonomy_level, support_mode, risk_level, opened_at, updated_at, closed_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?11, NULL)
        "#,
        params![
            case_id,
            ticket_key,
            label_assignment.label,
            bundle.label,
            bundle.bundle_version,
            state.as_str(),
            effective_control.approval_mode,
            effective_control.autonomy_level,
            bundle.support_mode,
            risk_level,
            now,
        ],
    )?;
    let artifact = build_dry_run_artifact(
        &ticket,
        &label_assignment,
        &bundle,
        &effective_control,
        &knowledge_load,
        understanding,
    );
    let dry_run_id = format!("dry-run:{}:{}", case_id, stable_digest(&now));
    conn.execute(
        r#"
        INSERT INTO ticket_dry_runs (
            dry_run_id, case_id, ticket_key, label, bundle_label, bundle_version, artifact_json, created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
        "#,
        params![
            dry_run_id,
            case_id,
            ticket_key,
            label_assignment.label,
            bundle.label,
            bundle.bundle_version,
            serde_json::to_string(&artifact)?,
            now,
        ],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key,
            case_id: Some(&case_id),
            actor_type: "control_plane",
            action_type: "label_contract_resolution",
            label: Some(&label_assignment.label),
            bundle_label: Some(&bundle.label),
            bundle_version: Some(bundle.bundle_version),
            details: json!({
                "runbook_id": bundle.runbook_id,
                "runbook_version": bundle.runbook_version,
                "policy_id": bundle.policy_id,
                "policy_version": bundle.policy_version,
                "requested_approval_mode": bundle.approval_mode,
                "requested_autonomy_level": bundle.autonomy_level,
                "effective_approval_mode": effective_control.approval_mode,
                "effective_autonomy_level": effective_control.autonomy_level,
                "grant": effective_control.grant.as_ref().map(|grant| {
                    json!({
                        "label": grant.label,
                        "grant_version": grant.grant_version,
                        "bundle_version": grant.bundle_version,
                        "approval_mode": grant.approval_mode,
                        "autonomy_level": grant.autonomy_level,
                        "approved_by": grant.approved_by,
                        "source_candidate_id": grant.source_candidate_id,
                    })
                }),
            }),
        },
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key,
            case_id: Some(&case_id),
            actor_type: "dry_run_engine",
            action_type: "dry_run_record",
            label: Some(&label_assignment.label),
            bundle_label: Some(&bundle.label),
            bundle_version: Some(bundle.bundle_version),
            details: artifact.clone(),
        },
    )?;
    load_latest_dry_run_for_case(root, &case_id)?.context("failed to load dry run after creation")
}

pub(super) fn build_dry_run_artifact(
    ticket: &TicketItemView,
    label_assignment: &TicketLabelAssignmentView,
    bundle: &ControlBundleView,
    effective_control: &EffectiveControlResolution,
    knowledge_load: &TicketKnowledgeLoadView,
    understanding: Option<&str>,
) -> Value {
    let actions = bundle
        .execution_actions
        .iter()
        .map(|action| {
            let execution_mode = if matches!(action.as_str(), "observe" | "analyze") {
                "executed_in_dry_run"
            } else {
                "simulated_only"
            };
            json!({
                "action_class": action,
                "execution_mode": execution_mode,
                "rationale": action_rationale(action),
            })
        })
        .collect::<Vec<_>>();
    json!({
        "ticket_understanding": understanding
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("{} [{}]", ticket.title.trim(), ticket.remote_status.trim())),
        "ticket_key": ticket.ticket_key,
        "knowledge_load": {
            "load_id": knowledge_load.load_id,
            "status": knowledge_load.status,
            "domains": knowledge_load.domains,
            "gap_domains": knowledge_load.gap_domains,
            "entries": knowledge_load.loaded_entries.iter().map(|entry| {
                json!({
                    "domain": entry.domain,
                    "knowledge_key": entry.knowledge_key,
                    "title": entry.title,
                    "summary": entry.summary,
                    "status": entry.status,
                })
            }).collect::<Vec<_>>(),
        },
        "bound_label": label_assignment.label,
        "runbook": {
            "id": bundle.runbook_id,
            "version": bundle.runbook_version,
        },
        "policy": {
            "id": bundle.policy_id,
            "version": bundle.policy_version,
            "approval_mode": effective_control.approval_mode,
            "autonomy_level": effective_control.autonomy_level,
            "support_mode": bundle.support_mode,
            "verification_profile_id": bundle.verification_profile_id,
            "writeback_profile_id": bundle.writeback_profile_id,
        },
        "requested_control": {
            "approval_mode": bundle.approval_mode,
            "autonomy_level": bundle.autonomy_level,
        },
        "autonomy_grant": effective_control.grant.as_ref().map(|grant| {
            json!({
                "grant_version": grant.grant_version,
                "bundle_version": grant.bundle_version,
                "approval_mode": grant.approval_mode,
                "autonomy_level": grant.autonomy_level,
                "approved_by": grant.approved_by,
                "source_candidate_id": grant.source_candidate_id,
            })
        }),
        "planned_actions": actions,
        "executed_now": ["observe", "analyze"],
        "simulated_only": bundle.execution_actions.iter().filter(|item| !matches!(item.as_str(), "observe" | "analyze")).cloned().collect::<Vec<_>>(),
        "missing_approvals": effective_control.missing_approvals,
        "required_evidence": required_evidence_for_bundle(bundle),
    })
}

pub fn list_cases(
    root: &Path,
    ticket_key: Option<&str>,
    limit: usize,
) -> Result<Vec<TicketCaseView>> {
    let conn = open_ticket_db(root)?;
    list_cases_on_conn(&conn, ticket_key, limit)
}

pub(super) fn list_cases_on_conn(
    conn: &Connection,
    ticket_key: Option<&str>,
    limit: usize,
) -> Result<Vec<TicketCaseView>> {
    let sql = if ticket_key.is_some() {
        r#"
        SELECT case_id, ticket_key, label, bundle_label, bundle_version, state, approval_mode,
               autonomy_level, support_mode, risk_level, opened_at, updated_at, closed_at
        FROM ticket_cases
        WHERE ticket_key = ?1
        ORDER BY updated_at DESC
        LIMIT ?2
        "#
    } else {
        r#"
        SELECT case_id, ticket_key, label, bundle_label, bundle_version, state, approval_mode,
               autonomy_level, support_mode, risk_level, opened_at, updated_at, closed_at
        FROM ticket_cases
        ORDER BY updated_at DESC
        LIMIT ?1
        "#
    };
    let mut statement = conn.prepare(sql)?;
    let rows = if let Some(ticket_key) = ticket_key {
        statement.query_map(params![ticket_key, limit as i64], map_case_row)?
    } else {
        statement.query_map(params![limit as i64], map_case_row)?
    };
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub(super) fn load_case(root: &Path, case_id: &str) -> Result<Option<TicketCaseView>> {
    let conn = open_ticket_db(root)?;
    conn.query_row(
        r#"
        SELECT case_id, ticket_key, label, bundle_label, bundle_version, state, approval_mode,
               autonomy_level, support_mode, risk_level, opened_at, updated_at, closed_at
        FROM ticket_cases
        WHERE case_id = ?1
        LIMIT 1
        "#,
        params![case_id],
        map_case_row,
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub(super) fn load_ticket_clarification_request(
    root: &Path,
    clarification_id: &str,
) -> Result<Option<TicketClarificationRequestView>> {
    let conn = open_ticket_db(root)?;
    conn.query_row(
        r#"
        SELECT clarification_id, ticket_key, case_id, work_id, target_type, target_channel,
               question, missing_inputs_json, unblock_criteria, status, outbound_message_key,
               inbound_response_key, inbound_response_body, resume_state, created_by,
               created_at, updated_at, sent_at, resolved_at, metadata_json
        FROM ticket_clarification_requests
        WHERE clarification_id = ?1
        LIMIT 1
        "#,
        params![clarification_id],
        map_ticket_clarification_row,
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub(super) fn load_latest_dry_run_for_case(
    root: &Path,
    case_id: &str,
) -> Result<Option<DryRunRecordView>> {
    let conn = open_ticket_db(root)?;
    conn.query_row(
        r#"
        SELECT dry_run_id, case_id, ticket_key, label, bundle_label, bundle_version, artifact_json, created_at
        FROM ticket_dry_runs
        WHERE case_id = ?1
        ORDER BY created_at DESC
        LIMIT 1
        "#,
        params![case_id],
        |row| {
            Ok(DryRunRecordView {
                dry_run_id: row.get(0)?,
                case_id: row.get(1)?,
                ticket_key: row.get(2)?,
                label: row.get(3)?,
                bundle_label: row.get(4)?,
                bundle_version: row.get(5)?,
                artifact: parse_json_column(row.get::<_, String>(6)?),
                created_at: row.get(7)?,
            })
        },
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub(super) fn decide_case_approval(
    root: &Path,
    case_id: &str,
    status: &str,
    decided_by: &str,
    rationale: Option<&str>,
) -> Result<TicketCaseView> {
    let mut conn = open_ticket_db(root)?;
    let case = load_case(root, case_id)?.context("ticket case not found")?;
    let canonical_status = canonical_approval_status(status)?;
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO ticket_approvals (approval_id, case_id, status, decided_by, rationale, created_at)
        VALUES (?1, ?2, ?3, ?4, ?5, ?6)
        "#,
        params![
            format!("approval:{}:{}", case_id, stable_digest(&now)),
            case_id,
            canonical_status,
            decided_by.trim(),
            rationale.map(str::trim),
            now,
        ],
    )?;
    let next_state = if canonical_status == "approved" {
        TicketCaseState::Executable
    } else {
        TicketCaseState::Blocked
    };
    enforce_ticket_case_state_transition(
        &conn,
        &case,
        next_state.as_str(),
        "approver",
        "approval_decision",
    )?;
    conn.execute(
        "UPDATE ticket_cases SET state = ?2, updated_at = ?3 WHERE case_id = ?1",
        params![case_id, next_state.as_str(), now],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(case_id),
            actor_type: "approver",
            action_type: "approval_decision",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({
                "status": canonical_status,
                "decided_by": decided_by.trim(),
                "rationale": rationale.map(str::trim),
            }),
        },
    )?;
    load_case(root, case_id)?.context("failed to load case after approval decision")
}

pub(super) fn record_execution_action(
    root: &Path,
    case_id: &str,
    summary: &str,
) -> Result<TicketCaseView> {
    let mut conn = open_ticket_db(root)?;
    let case = load_case(root, case_id)?.context("ticket case not found")?;
    ensure_case_is_executable(&case)?;
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO ticket_execution_actions (
            action_id, case_id, ticket_key, summary, created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5)
        "#,
        params![
            format!("execution:{}:{}", case_id, stable_digest(&now)),
            case_id,
            case.ticket_key,
            summary.trim(),
            now,
        ],
    )?;
    let next_state = TicketCaseState::Executing;
    enforce_ticket_case_state_transition(
        &conn,
        &case,
        next_state.as_str(),
        "agent",
        "execution_case",
    )?;
    conn.execute(
        "UPDATE ticket_cases SET state = ?2, updated_at = ?3 WHERE case_id = ?1",
        params![case_id, next_state.as_str(), now],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(case_id),
            actor_type: "agent",
            action_type: "execution_case",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({"summary": summary.trim()}),
        },
    )?;
    load_case(root, case_id)?.context("failed to load case after execution action")
}

pub(super) fn record_verification(
    root: &Path,
    case_id: &str,
    status: &str,
    summary: Option<&str>,
) -> Result<TicketCaseView> {
    let mut conn = open_ticket_db(root)?;
    let case = load_case(root, case_id)?.context("ticket case not found")?;
    let canonical_status = canonical_verification_status(status)?;
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO ticket_verifications (
            verification_id, case_id, status, summary, created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5)
        "#,
        params![
            format!("verification:{}:{}", case_id, stable_digest(&now)),
            case_id,
            canonical_status,
            summary.map(str::trim),
            now,
        ],
    )?;
    let next_state = if canonical_status == "passed" {
        TicketCaseState::WritebackPending
    } else {
        TicketCaseState::Blocked
    };
    enforce_ticket_case_state_transition(
        &conn,
        &case,
        next_state.as_str(),
        "verification_engine",
        "verification_record",
    )?;
    conn.execute(
        "UPDATE ticket_cases SET state = ?2, updated_at = ?3 WHERE case_id = ?1",
        params![case_id, next_state.as_str(), now],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(case_id),
            actor_type: "verification_engine",
            action_type: "verification_record",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({
                "status": canonical_status,
                "summary": summary.map(str::trim),
            }),
        },
    )?;
    load_case(root, case_id)?.context("failed to load case after verification")
}

pub(super) fn create_ticket_clarification_request(
    root: &Path,
    input: TicketClarificationRequestInput,
) -> Result<TicketClarificationRequestView> {
    let mut conn = open_ticket_db(root)?;
    let case = match input.case_id.as_deref() {
        Some(case_id) => Some(load_case(root, case_id)?.context("ticket case not found")?),
        None => None,
    };
    let ticket_key = match (&case, input.ticket_key.as_deref()) {
        (Some(case), Some(ticket_key)) if case.ticket_key != ticket_key => {
            anyhow::bail!(
                "clarification ticket_key {} does not match case {} ticket_key {}",
                ticket_key,
                case.case_id,
                case.ticket_key
            );
        }
        (Some(case), _) => case.ticket_key.clone(),
        (None, Some(ticket_key)) => ticket_key.to_string(),
        (None, None) => anyhow::bail!("case_id or ticket_key is required for clarification"),
    };
    let ticket = load_ticket(root, &ticket_key)?.context("ticket not found for clarification")?;
    let target_type = canonical_clarification_target_type(&input.target_type)?;
    let target_channel = canonical_clarification_target_channel(&input.target_channel)?;
    let resume_state = canonical_clarification_resume_state(&input.resume_state)?;
    let question = input.question.trim();
    anyhow::ensure!(!question.is_empty(), "clarification question is required");
    let missing_inputs = normalize_clarification_inputs(input.missing_inputs);
    let now = now_iso_string();
    let clarification_id = format!(
        "clarification:{}:{}",
        ticket_key,
        stable_digest(&(question.to_string() + now.as_str()))
    );
    if let Some(case) = case.as_ref() {
        ensure_case_can_request_clarification(case)?;
        enforce_ticket_case_state_transition(
            &conn,
            case,
            "blocked_needs_clarification",
            "clarification_engine",
            "missing_info_request",
        )?;
    }
    conn.execute(
        r#"
        INSERT INTO ticket_clarification_requests (
            clarification_id, ticket_key, case_id, work_id, target_type, target_channel,
            question, missing_inputs_json, unblock_criteria, status, outbound_message_key,
            inbound_response_key, inbound_response_body, resume_state, created_by,
            metadata_json, created_at, updated_at, sent_at, resolved_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'draft', NULL, NULL, NULL, ?10, ?11, ?12, ?13, ?13, NULL, NULL)
        "#,
        params![
            clarification_id,
            ticket_key,
            case.as_ref().map(|case| case.case_id.as_str()),
            input.work_id.as_deref().map(str::trim),
            &target_type,
            &target_channel,
            question,
            serde_json::to_string(&missing_inputs)?,
            input.unblock_criteria.as_deref().map(str::trim),
            &resume_state,
            input.created_by.trim(),
            serde_json::to_string(&input.metadata)?,
            now,
        ],
    )?;
    if let Some(case) = case.as_ref() {
        conn.execute(
            "UPDATE ticket_cases SET state = 'blocked_needs_clarification', updated_at = ?2 WHERE case_id = ?1",
            params![case.case_id, now],
        )?;
    }
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &ticket.ticket_key,
            case_id: case.as_ref().map(|case| case.case_id.as_str()),
            actor_type: "clarification_engine",
            action_type: "clarification_requested",
            label: case.as_ref().map(|case| case.label.as_str()),
            bundle_label: case.as_ref().map(|case| case.bundle_label.as_str()),
            bundle_version: case.as_ref().map(|case| case.bundle_version),
            details: json!({
                "clarification_id": clarification_id,
                "target_type": target_type,
                "target_channel": target_channel,
                "question": question,
                "missing_inputs": missing_inputs,
                "unblock_criteria": input.unblock_criteria.as_deref().map(str::trim),
            }),
        },
    )?;
    load_ticket_clarification_request(root, &clarification_id)?
        .context("failed to load ticket clarification after create")
}

pub(super) fn publish_ticket_clarification_request(
    root: &Path,
    clarification_id: &str,
    reviewed_by: &str,
    review_summary: &str,
) -> Result<TicketClarificationRequestView> {
    let mut conn = open_ticket_db(root)?;
    let clarification = load_ticket_clarification_request(root, clarification_id)?
        .context("ticket clarification not found")?;
    anyhow::ensure!(
        matches!(clarification.status.as_str(), "draft" | "send_failed"),
        "clarification {} cannot be published from status {}",
        clarification.clarification_id,
        clarification.status
    );
    anyhow::ensure!(
        !reviewed_by.trim().is_empty() && !review_summary.trim().is_empty(),
        "publishing a clarification requires reviewed_by and review_summary"
    );
    anyhow::ensure!(
        clarification.target_type == "requester" && clarification.target_channel == "ticket",
        "automatic clarification publish currently supports requester ticket comments only"
    );
    let ticket = load_ticket(root, &clarification.ticket_key)?
        .context("ticket not found for clarification")?;
    let Some(adapter) = ticket_adapters::adapter_for_system(&ticket.source_system) else {
        anyhow::bail!(
            "unsupported ticket system for clarification publish: {}",
            ticket.source_system
        );
    };
    let capabilities = adapter.capabilities();
    anyhow::ensure!(
        capabilities.can_comment_writeback && capabilities.can_public_comments,
        "ticket system {} does not support public clarification comments",
        ticket.source_system
    );
    let result = match adapter.writeback_comment(
        root,
        ticket_protocol::TicketCommentWritebackRequest {
            remote_ticket_id: &ticket.remote_ticket_id,
            body: &clarification.question,
            internal: false,
        },
    ) {
        Ok(result) => result,
        Err(err) => {
            let error = err.to_string();
            let now = now_iso_string();
            conn.execute(
                r#"
                UPDATE ticket_clarification_requests
                SET status = 'send_failed',
                    metadata_json = ?2,
                    updated_at = ?3
                WHERE clarification_id = ?1
                "#,
                params![
                    clarification_id,
                    serde_json::to_string(&json!({
                        "previous": clarification.metadata,
                        "send_error": collapse_inline(&error, 1000),
                    }))?,
                    now,
                ],
            )?;
            anyhow::bail!("{}", error);
        }
    };
    mark_remote_events_outbound(root, &ticket.source_system, &result.remote_event_ids)?;
    if let Err(err) = sync_ticket_system(root, &ticket.source_system) {
        let _ = record_ticket_sync_failure(root, &ticket.source_system, &err.to_string());
    }
    let response_baseline_event_keys =
        list_inbound_ticket_event_keys(&conn, &clarification.ticket_key)?;
    let now = now_iso_string();
    let writeback_id = format!(
        "clarification-writeback:{}:{}",
        clarification_id,
        stable_digest(&now)
    );
    conn.execute(
        r#"
        INSERT INTO ticket_writebacks (
            writeback_id, case_id, ticket_key, operation, payload_json, status, created_at
        ) VALUES (?1, ?2, ?3, 'clarification_request', ?4, 'ok', ?5)
        "#,
        params![
            writeback_id,
            clarification.case_id.as_deref().unwrap_or(""),
            &clarification.ticket_key,
            serde_json::to_string(&json!({
                "clarification_id": clarification.clarification_id,
                "body": clarification.question,
                "reviewed_by": reviewed_by.trim(),
                "review_summary": review_summary.trim(),
                "remote_event_ids": result.remote_event_ids.clone(),
            }))?,
            now,
        ],
    )?;
    let outbound_message_key = result
        .remote_event_ids
        .first()
        .cloned()
        .unwrap_or_else(|| writeback_id.clone());
    conn.execute(
        r#"
        UPDATE ticket_clarification_requests
        SET status = 'waiting_for_response',
            outbound_message_key = ?2,
            metadata_json = ?3,
            updated_at = ?4,
            sent_at = ?4
        WHERE clarification_id = ?1
        "#,
        params![
            clarification_id,
            outbound_message_key,
            serde_json::to_string(&json!({
                "previous": clarification.metadata.clone(),
                "reviewed_by": reviewed_by.trim(),
                "review_summary": review_summary.trim(),
                "writeback_id": writeback_id,
                "response_baseline_event_keys": response_baseline_event_keys,
            }))?,
            now,
        ],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &ticket.ticket_key,
            case_id: clarification.case_id.as_deref(),
            actor_type: "clarification_engine",
            action_type: "clarification_published",
            label: None,
            bundle_label: None,
            bundle_version: None,
            details: json!({
                "clarification_id": clarification_id,
                "reviewed_by": reviewed_by.trim(),
                "review_summary": review_summary.trim(),
                "outbound_message_key": outbound_message_key,
            }),
        },
    )?;
    load_ticket_clarification_request(root, clarification_id)?
        .context("failed to load ticket clarification after publish")
}

pub(super) fn resolve_ticket_clarification_request(
    root: &Path,
    clarification_id: &str,
    response_key: &str,
    response_body: Option<&str>,
    resolved_by: &str,
) -> Result<TicketClarificationRequestView> {
    let mut conn = open_ticket_db(root)?;
    let clarification = load_ticket_clarification_request(root, clarification_id)?
        .context("ticket clarification not found")?;
    let now = now_iso_string();
    conn.execute(
        r#"
        UPDATE ticket_clarification_requests
        SET status = 'resolved',
            inbound_response_key = ?2,
            inbound_response_body = ?3,
            updated_at = ?4,
            resolved_at = ?4,
            metadata_json = ?5
        WHERE clarification_id = ?1
        "#,
        params![
            clarification_id,
            response_key.trim(),
            response_body.map(str::trim),
            now,
            serde_json::to_string(&json!({
                "previous": clarification.metadata.clone(),
                "resolved_by": resolved_by.trim(),
            }))?,
        ],
    )?;
    if let Some(case_id) = clarification.case_id.as_deref() {
        if let Some(case) = load_case(root, case_id)? {
            if matches!(
                case.state.as_str(),
                "blocked" | "blocked_needs_clarification"
            ) {
                let resume_state = TicketCaseState::parse(&clarification.resume_state)?;
                enforce_ticket_case_state_transition(
                    &conn,
                    &case,
                    resume_state.as_str(),
                    "clarification_engine",
                    "clarification_resolved",
                )?;
                conn.execute(
                    "UPDATE ticket_cases SET state = ?2, updated_at = ?3 WHERE case_id = ?1",
                    params![case_id, resume_state.as_str(), now],
                )?;
            }
        }
    }
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &clarification.ticket_key,
            case_id: clarification.case_id.as_deref(),
            actor_type: "clarification_engine",
            action_type: "clarification_resolved",
            label: None,
            bundle_label: None,
            bundle_version: None,
            details: json!({
                "clarification_id": clarification_id,
                "response_key": response_key.trim(),
                "resolved_by": resolved_by.trim(),
            }),
        },
    )?;
    load_ticket_clarification_request(root, clarification_id)?
        .context("failed to load ticket clarification after resolve")
}

pub(super) fn resolve_waiting_clarifications_from_inbound_events(
    root: &Path,
    system: &str,
) -> Result<usize> {
    let conn = open_ticket_db(root)?;
    let mut statement = conn.prepare(
        r#"
        SELECT c.clarification_id, c.ticket_key, COALESCE(c.sent_at, c.created_at), c.metadata_json
        FROM ticket_clarification_requests c
        JOIN ticket_items t ON t.ticket_key = c.ticket_key
        WHERE t.source_system = ?1
          AND c.status = 'waiting_for_response'
        ORDER BY c.created_at ASC
        "#,
    )?;
    let rows = statement.query_map(params![system], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    let waiting = rows
        .collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)?;
    drop(statement);
    drop(conn);

    let mut resolved = 0usize;
    for (clarification_id, ticket_key, since, metadata_raw) in waiting {
        let response_baseline_event_keys =
            clarification_response_baseline_event_keys(&parse_json_or_empty(&metadata_raw));
        let conn = open_ticket_db(root)?;
        let mut response_statement = conn.prepare(
            r#"
                SELECT event_key, body_text
                FROM ticket_events
                WHERE ticket_key = ?1
                  AND direction = 'inbound'
                  AND observed_at >= ?2
                  AND trim(body_text) <> ''
                ORDER BY external_created_at ASC, observed_at ASC, event_key ASC
                "#,
        )?;
        let response_rows = response_statement.query_map(params![ticket_key, since], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let response = response_rows
            .collect::<rusqlite::Result<Vec<_>>>()
            .map_err(anyhow::Error::from)?
            .into_iter()
            .find(|(event_key, _)| !response_baseline_event_keys.contains(event_key));
        drop(response_statement);
        drop(conn);
        if let Some((event_key, body_text)) = response {
            resolve_ticket_clarification_request(
                root,
                &clarification_id,
                &event_key,
                Some(&body_text),
                "ticket-sync",
            )?;
            resolved += 1;
        }
    }
    Ok(resolved)
}

pub(super) fn list_inbound_ticket_event_keys(
    conn: &Connection,
    ticket_key: &str,
) -> Result<Vec<String>> {
    let mut statement = conn.prepare(
        r#"
        SELECT event_key
        FROM ticket_events
        WHERE ticket_key = ?1
          AND direction = 'inbound'
        ORDER BY external_created_at ASC, observed_at ASC, event_key ASC
        "#,
    )?;
    let rows = statement.query_map(params![ticket_key], |row| row.get::<_, String>(0))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub(super) fn clarification_response_baseline_event_keys(metadata: &Value) -> BTreeSet<String> {
    metadata
        .get("response_baseline_event_keys")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(str::trim)
                .filter(|item| !item.is_empty())
                .map(ToOwned::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

pub(super) fn writeback_comment(
    root: &Path,
    case_id: &str,
    body: &str,
    internal: bool,
) -> Result<TicketCaseView> {
    let mut conn = open_ticket_db(root)?;
    let case = load_case(root, case_id)?.context("ticket case not found")?;
    ensure_case_ready_for_writeback(&case)?;
    let ticket = load_ticket(root, &case.ticket_key)?.context("ticket not found for case")?;
    let Some(adapter) = ticket_adapters::adapter_for_system(&ticket.source_system) else {
        anyhow::bail!(
            "unsupported ticket system for writeback: {}",
            ticket.source_system
        );
    };
    let capabilities = adapter.capabilities();
    if !capabilities.can_comment_writeback {
        anyhow::bail!(
            "ticket system {} does not support comment writeback",
            ticket.source_system
        );
    }
    if internal && !capabilities.can_internal_comments {
        anyhow::bail!(
            "ticket system {} does not support internal comments",
            ticket.source_system
        );
    }
    if !internal && !capabilities.can_public_comments {
        anyhow::bail!(
            "ticket system {} does not support public comments",
            ticket.source_system
        );
    }
    let result = match adapter.writeback_comment(
        root,
        ticket_protocol::TicketCommentWritebackRequest {
            remote_ticket_id: &ticket.remote_ticket_id,
            body,
            internal,
        },
    ) {
        Ok(result) => result,
        Err(err) => {
            let error = err.to_string();
            record_failed_writeback(
                &mut conn,
                &case,
                "comment",
                json!({
                    "body": body.trim(),
                    "internal": internal,
                    "remote_ticket_id": ticket.remote_ticket_id.clone(),
                    "source_system": ticket.source_system.clone(),
                }),
                &error,
            )?;
            anyhow::bail!("{}", error);
        }
    };
    mark_remote_events_outbound(root, &ticket.source_system, &result.remote_event_ids)?;
    if let Err(err) = sync_ticket_system(root, &ticket.source_system) {
        let _ = record_ticket_sync_failure(root, &ticket.source_system, &err.to_string());
    }
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO ticket_writebacks (
            writeback_id, case_id, ticket_key, operation, payload_json, status, created_at
        ) VALUES (?1, ?2, ?3, 'comment', ?4, 'ok', ?5)
        "#,
        params![
            format!("writeback:{}:{}", case_id, stable_digest(&now)),
            case_id,
            case.ticket_key,
            serde_json::to_string(&json!({
                "body": body.trim(),
                "internal": internal,
                "remote_event_ids": result.remote_event_ids,
            }))?,
            now,
        ],
    )?;
    conn.execute(
        "UPDATE ticket_cases SET updated_at = ?2 WHERE case_id = ?1",
        params![case_id, now],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(case_id),
            actor_type: "writeback_engine",
            action_type: "writeback_record",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({
                "operation": "comment",
                "body": body.trim(),
                "internal": internal,
            }),
        },
    )?;
    load_case(root, case_id)?.context("failed to load case after writeback")
}

pub(super) fn writeback_transition(
    root: &Path,
    case_id: &str,
    state: &str,
    note_body: Option<&str>,
    internal_note: bool,
) -> Result<TicketCaseView> {
    let mut conn = open_ticket_db(root)?;
    let case = load_case(root, case_id)?.context("ticket case not found")?;
    ensure_case_ready_for_writeback(&case)?;
    let ticket = load_ticket(root, &case.ticket_key)?.context("ticket not found for case")?;
    let Some(adapter) = ticket_adapters::adapter_for_system(&ticket.source_system) else {
        anyhow::bail!(
            "unsupported ticket system for writeback: {}",
            ticket.source_system
        );
    };
    let capabilities = adapter.capabilities();
    if !capabilities.can_transition_writeback {
        anyhow::bail!(
            "ticket system {} does not support state transitions",
            ticket.source_system
        );
    }
    if internal_note && !capabilities.can_internal_comments {
        anyhow::bail!(
            "ticket system {} does not support internal notes on transitions",
            ticket.source_system
        );
    }
    if note_body.is_some() && !internal_note && !capabilities.can_public_comments {
        anyhow::bail!(
            "ticket system {} does not support public transition notes",
            ticket.source_system
        );
    }
    enforce_ticket_case_close_transition(&conn, &case, "writeback_engine")?;
    let result = match adapter.writeback_transition(
        root,
        ticket_protocol::TicketTransitionWritebackRequest {
            remote_ticket_id: &ticket.remote_ticket_id,
            state,
            note_body,
            internal_note,
            control_note: None,
        },
    ) {
        Ok(result) => result,
        Err(err) => {
            let error = err.to_string();
            record_failed_writeback(
                &mut conn,
                &case,
                "transition",
                json!({
                    "state": state.trim(),
                    "note_body": note_body.map(str::trim),
                    "internal_note": internal_note,
                    "remote_ticket_id": ticket.remote_ticket_id.clone(),
                    "source_system": ticket.source_system.clone(),
                }),
                &error,
            )?;
            anyhow::bail!("{}", error);
        }
    };
    mark_remote_events_outbound(root, &ticket.source_system, &result.remote_event_ids)?;
    if let Err(err) = sync_ticket_system(root, &ticket.source_system) {
        let _ = record_ticket_sync_failure(root, &ticket.source_system, &err.to_string());
    }
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO ticket_writebacks (
            writeback_id, case_id, ticket_key, operation, payload_json, status, created_at
        ) VALUES (?1, ?2, ?3, 'transition', ?4, 'ok', ?5)
        "#,
        params![
            format!(
                "writeback:{}:{}",
                case_id,
                stable_digest(&(state.to_string() + now.as_str()))
            ),
            case_id,
            case.ticket_key,
            serde_json::to_string(&json!({
                "state": state.trim(),
                "note_body": note_body.map(str::trim),
                "internal_note": internal_note,
                "remote_event_ids": result.remote_event_ids,
            }))?,
            now,
        ],
    )?;
    enforce_ticket_case_close_transition(&conn, &case, "writeback_engine")?;
    conn.execute(
        "UPDATE ticket_cases SET state = 'closed', updated_at = ?2, closed_at = ?2 WHERE case_id = ?1",
        params![case_id, now],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(case_id),
            actor_type: "writeback_engine",
            action_type: "writeback_record",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({
                "operation": "transition",
                "state": state.trim(),
                "note_body": note_body.map(str::trim),
                "internal_note": internal_note,
            }),
        },
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(case_id),
            actor_type: "control_plane",
            action_type: "case_closed",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({"reason": "writeback transition completed"}),
        },
    )?;
    load_case(root, case_id)?.context("failed to load case after transition writeback")
}

pub(super) fn close_case(
    root: &Path,
    case_id: &str,
    summary: Option<&str>,
) -> Result<TicketCaseView> {
    let mut conn = open_ticket_db(root)?;
    let case = load_case(root, case_id)?.context("ticket case not found")?;
    enforce_ticket_case_close_transition(&conn, &case, "control_plane")?;
    let now = now_iso_string();
    conn.execute(
        "UPDATE ticket_cases SET state = 'closed', updated_at = ?2, closed_at = ?2 WHERE case_id = ?1",
        params![case_id, now],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(case_id),
            actor_type: "control_plane",
            action_type: "case_closed",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({"summary": summary.map(str::trim)}),
        },
    )?;
    load_case(root, case_id)?.context("failed to load case after close")
}

pub(crate) fn run_business_os_ticket_command(
    root: &Path,
    command_type: &str,
    payload: &Value,
) -> Result<Value> {
    match command_type {
        "ctox.ticket.local.create" => {
            let title = required_payload_string(payload, "title")?;
            let body = payload_string(payload, "body")
                .or_else(|| payload_string(payload, "body_text"))
                .unwrap_or_default();
            let record = crate::mission::ticket_local_native::create_local_ticket(
                root,
                &title,
                &body,
                payload_string(payload, "status").as_deref(),
                payload_string(payload, "priority").as_deref(),
            )?;
            sync_ticket_system(root, "local")?;
            Ok(json!({
                "ticket": record,
                "ticket_key": record.ticket_id,
                "source_system": "local",
            }))
        }
        "ctox.ticket.local.comment" => {
            let ticket_id = required_payload_string(payload, "ticket_id")
                .or_else(|_| required_payload_string(payload, "ticket_key"))?;
            let body = required_payload_string(payload, "body")?;
            let event =
                crate::mission::ticket_local_native::add_local_comment(root, &ticket_id, &body)?;
            sync_ticket_system(root, "local")?;
            Ok(json!({
                "event": event,
                "ticket_key": ticket_id,
                "source_system": "local",
            }))
        }
        "ctox.ticket.local.transition" => {
            let ticket_id = required_payload_string(payload, "ticket_id")
                .or_else(|_| required_payload_string(payload, "ticket_key"))?;
            let status = required_payload_string(payload, "status")
                .or_else(|_| required_payload_string(payload, "state"))?;
            let record = crate::mission::ticket_local_native::transition_local_ticket(
                root, &ticket_id, &status,
            )?;
            sync_ticket_system(root, "local")?;
            Ok(json!({
                "ticket": record,
                "ticket_key": ticket_id,
                "source_system": "local",
            }))
        }
        "ctox.ticket.approve" => {
            let case_id = required_payload_string(payload, "case_id")?;
            let status = required_payload_string(payload, "status")?;
            let decided_by =
                payload_string(payload, "decided_by").unwrap_or_else(|| "owner".to_string());
            let case = decide_case_approval(
                root,
                &case_id,
                &status,
                &decided_by,
                payload_string(payload, "rationale").as_deref(),
            )?;
            Ok(json!({ "case": case, "case_id": case.case_id, "ticket_key": case.ticket_key }))
        }
        "ctox.ticket.execute" => {
            let case_id = required_payload_string(payload, "case_id")?;
            let summary = required_payload_string(payload, "summary")?;
            let case = record_execution_action(root, &case_id, &summary)?;
            Ok(json!({ "case": case, "case_id": case.case_id, "ticket_key": case.ticket_key }))
        }
        "ctox.ticket.verify" => {
            let case_id = required_payload_string(payload, "case_id")?;
            let status = required_payload_string(payload, "status")?;
            let case = record_verification(
                root,
                &case_id,
                &status,
                payload_string(payload, "summary").as_deref(),
            )?;
            Ok(json!({ "case": case, "case_id": case.case_id, "ticket_key": case.ticket_key }))
        }
        "ctox.ticket.request_clarification" => {
            let case_id = payload_string(payload, "case_id");
            let ticket_key = payload_string(payload, "ticket_key");
            let question = required_payload_string(payload, "question")?;
            let missing_inputs = payload
                .get("missing_inputs")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(Value::as_str)
                        .map(ToOwned::to_owned)
                        .collect::<Vec<_>>()
                })
                .unwrap_or_else(|| {
                    payload_string(payload, "missing_inputs_csv")
                        .map(|value| parse_domain_csv(&value))
                        .unwrap_or_default()
                });
            let clarification = create_ticket_clarification_request(
                root,
                TicketClarificationRequestInput {
                    case_id,
                    ticket_key,
                    work_id: payload_string(payload, "work_id"),
                    target_type: payload_string(payload, "target_type")
                        .unwrap_or_else(|| "requester".to_string()),
                    target_channel: payload_string(payload, "target_channel")
                        .unwrap_or_else(|| "ticket".to_string()),
                    question,
                    missing_inputs,
                    unblock_criteria: payload_string(payload, "unblock_criteria"),
                    resume_state: payload_string(payload, "resume_state")
                        .unwrap_or_else(|| "executable".to_string()),
                    created_by: payload_string(payload, "created_by")
                        .unwrap_or_else(|| "business-os".to_string()),
                    metadata: payload
                        .get("metadata")
                        .cloned()
                        .unwrap_or_else(|| json!({})),
                },
            )?;
            let clarification_id = clarification.clarification_id.clone();
            let case_id = clarification.case_id.clone();
            let ticket_key = clarification.ticket_key.clone();
            Ok(json!({
                "clarification": clarification,
                "clarification_id": clarification_id,
                "case_id": case_id,
                "ticket_key": ticket_key
            }))
        }
        "ctox.ticket.publish_clarification" => {
            let clarification_id = required_payload_string(payload, "clarification_id")?;
            let reviewed_by =
                payload_string(payload, "reviewed_by").unwrap_or_else(|| "business-os".to_string());
            let review_summary = payload_string(payload, "review_summary")
                .unwrap_or_else(|| "Clarification question reviewed for this ticket.".to_string());
            let clarification = publish_ticket_clarification_request(
                root,
                &clarification_id,
                &reviewed_by,
                &review_summary,
            )?;
            let case_id = clarification.case_id.clone();
            let ticket_key = clarification.ticket_key.clone();
            Ok(json!({
                "clarification": clarification,
                "clarification_id": clarification_id,
                "case_id": case_id,
                "ticket_key": ticket_key
            }))
        }
        "ctox.ticket.resolve_clarification" => {
            let clarification_id = required_payload_string(payload, "clarification_id")?;
            let response_key = required_payload_string(payload, "response_key")?;
            let clarification = resolve_ticket_clarification_request(
                root,
                &clarification_id,
                &response_key,
                payload_string(payload, "body").as_deref(),
                payload_string(payload, "resolved_by")
                    .as_deref()
                    .unwrap_or("business-os"),
            )?;
            let case_id = clarification.case_id.clone();
            let ticket_key = clarification.ticket_key.clone();
            Ok(json!({
                "clarification": clarification,
                "clarification_id": clarification_id,
                "case_id": case_id,
                "ticket_key": ticket_key
            }))
        }
        "ctox.ticket.writeback_comment" => {
            let case_id = required_payload_string(payload, "case_id")?;
            let body = required_payload_string(payload, "body")?;
            let case = writeback_comment(
                root,
                &case_id,
                &body,
                payload_bool(payload, "internal").unwrap_or(false),
            )?;
            Ok(json!({ "case": case, "case_id": case.case_id, "ticket_key": case.ticket_key }))
        }
        "ctox.ticket.writeback_transition" => {
            let case_id = required_payload_string(payload, "case_id")?;
            let state = required_payload_string(payload, "state")?;
            let case = writeback_transition(
                root,
                &case_id,
                &state,
                payload_string(payload, "body").as_deref(),
                payload_bool(payload, "internal").unwrap_or(false),
            )?;
            Ok(json!({ "case": case, "case_id": case.case_id, "ticket_key": case.ticket_key }))
        }
        "ctox.ticket.close" => {
            let case_id = required_payload_string(payload, "case_id")?;
            let case = close_case(
                root,
                &case_id,
                payload_string(payload, "summary").as_deref(),
            )?;
            Ok(json!({ "case": case, "case_id": case.case_id, "ticket_key": case.ticket_key }))
        }
        other => anyhow::bail!("unsupported Business OS ticket command: {other}"),
    }
}

pub(super) fn required_payload_string(payload: &Value, key: &str) -> Result<String> {
    payload_string(payload, key)
        .filter(|value| !value.trim().is_empty())
        .with_context(|| format!("{key} is required"))
}

pub(super) fn payload_string(payload: &Value, key: &str) -> Option<String> {
    payload
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(str::to_string)
}

pub(super) fn payload_bool(payload: &Value, key: &str) -> Option<bool> {
    payload.get(key).and_then(Value::as_bool)
}

pub(super) fn enforce_ticket_case_close_transition(
    conn: &Connection,
    case: &TicketCaseView,
    actor: &str,
) -> Result<()> {
    let verification_id = latest_passed_ticket_verification_id(conn, &case.case_id)?;
    let from_state = ticket_case_core_state(&case.state)?;
    let mut metadata = BTreeMap::new();
    metadata.insert("ticket_key".to_string(), case.ticket_key.clone());
    metadata.insert("label".to_string(), case.label.clone());
    metadata.insert("support_mode".to_string(), case.support_mode.clone());
    metadata.insert("owner_visible_completion".to_string(), "true".to_string());
    metadata.insert("completion_review_required".to_string(), "true".to_string());
    metadata.insert("completion_review_verdict".to_string(), "pass".to_string());
    metadata.insert(
        "reviewed_work_terminal_success".to_string(),
        "true".to_string(),
    );

    enforce_core_transition(
        conn,
        &CoreTransitionRequest {
            entity_type: CoreEntityType::Ticket,
            entity_id: case.case_id.clone(),
            lane: RuntimeLane::P2MissionDelivery,
            from_state,
            to_state: CoreState::Closed,
            event: CoreEvent::Close,
            actor: actor.to_string(),
            evidence: CoreEvidenceRefs {
                verification_id,
                review_audit_key: latest_ticket_review_audit_key(conn, &case.case_id)?,
                ..CoreEvidenceRefs::default()
            },
            metadata,
        },
    )?;
    Ok(())
}

pub(super) fn enforce_ticket_case_create_transition(
    conn: &Connection,
    case_id: &str,
    ticket_key: &str,
    state: &str,
    label: &str,
    support_mode: &str,
    actor: &str,
    reason: &str,
) -> Result<()> {
    let to_core_state = ticket_case_core_state(state)?;
    let mut metadata = BTreeMap::new();
    metadata.insert("ticket_key".to_string(), ticket_key.to_string());
    metadata.insert("label".to_string(), label.to_string());
    metadata.insert("support_mode".to_string(), support_mode.to_string());
    metadata.insert("from_case_state".to_string(), "created".to_string());
    metadata.insert("to_case_state".to_string(), state.to_string());
    metadata.insert("reason".to_string(), reason.to_string());
    enforce_core_transition(
        conn,
        &CoreTransitionRequest {
            entity_type: CoreEntityType::Ticket,
            entity_id: case_id.to_string(),
            lane: RuntimeLane::P2MissionDelivery,
            from_state: CoreState::Created,
            to_state: to_core_state,
            event: ticket_case_core_event(state)?,
            actor: actor.to_string(),
            evidence: CoreEvidenceRefs::default(),
            metadata,
        },
    )?;
    Ok(())
}

pub(super) fn enforce_ticket_case_state_transition(
    conn: &Connection,
    case: &TicketCaseView,
    to_state: &str,
    actor: &str,
    reason: &str,
) -> Result<()> {
    let from_state = ticket_case_core_state(&case.state)?;
    let to_core_state = ticket_case_core_state(to_state)?;
    let mut metadata = BTreeMap::new();
    metadata.insert("ticket_key".to_string(), case.ticket_key.clone());
    metadata.insert("label".to_string(), case.label.clone());
    metadata.insert("support_mode".to_string(), case.support_mode.clone());
    metadata.insert("from_case_state".to_string(), case.state.clone());
    metadata.insert("to_case_state".to_string(), to_state.to_string());
    metadata.insert("reason".to_string(), reason.to_string());
    enforce_core_transition(
        conn,
        &CoreTransitionRequest {
            entity_type: CoreEntityType::Ticket,
            entity_id: case.case_id.clone(),
            lane: RuntimeLane::P2MissionDelivery,
            from_state,
            to_state: to_core_state,
            event: ticket_case_core_event(to_state)?,
            actor: actor.to_string(),
            evidence: CoreEvidenceRefs::default(),
            metadata,
        },
    )?;
    Ok(())
}

pub(super) fn latest_passed_ticket_verification_id(
    conn: &Connection,
    case_id: &str,
) -> Result<Option<String>> {
    conn.query_row(
        r#"
        SELECT verification_id
        FROM ticket_verifications
        WHERE case_id = ?1 AND status = 'passed'
        ORDER BY created_at DESC
        LIMIT 1
        "#,
        params![case_id],
        |row| row.get(0),
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub(super) fn latest_ticket_review_audit_key(
    conn: &Connection,
    case_id: &str,
) -> Result<Option<String>> {
    conn.query_row(
        r#"
        SELECT audit_id
        FROM ticket_audit_log
        WHERE case_id = ?1
          AND action_type IN ('source_skill_review_note', 'approval_decision', 'verification_record')
        ORDER BY created_at DESC
        LIMIT 1
        "#,
        params![case_id],
        |row| row.get(0),
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub(super) fn ticket_case_core_state(raw: &str) -> Result<CoreState> {
    Ok(TicketCaseState::parse(raw)?.core_state())
}

pub(super) fn ticket_case_core_event(raw: &str) -> Result<CoreEvent> {
    Ok(TicketCaseState::parse(raw)?.core_event())
}

pub(super) fn create_learning_candidate(
    root: &Path,
    case_id: &str,
    summary: &str,
    proposed_actions_override: Option<&[String]>,
    evidence_override: Option<Value>,
) -> Result<LearningCandidateView> {
    let mut conn = open_ticket_db(root)?;
    let case = load_case(root, case_id)?.context("ticket case not found")?;
    let dry_run = load_latest_dry_run_for_case(root, case_id)?
        .context("dry run is required before creating a learning candidate")?;
    let proposed_actions = proposed_actions_override
        .map(|items| items.to_vec())
        .unwrap_or_else(|| {
            dry_run
                .artifact
                .get("planned_actions")
                .and_then(Value::as_array)
                .map(|items| {
                    items
                        .iter()
                        .filter_map(|item| item.get("action_class").and_then(Value::as_str))
                        .map(ToOwned::to_owned)
                        .collect::<Vec<_>>()
                })
                .filter(|items| !items.is_empty())
                .unwrap_or_else(default_execution_actions)
        });
    let evidence = evidence_override.unwrap_or_else(|| {
        json!({
            "case_state": case.state,
            "dry_run_id": dry_run.dry_run_id,
            "dry_run_artifact": dry_run.artifact,
        })
    });
    let now = now_iso_string();
    let candidate_id = format!("candidate:{}:{}", case_id, stable_digest(&now));
    conn.execute(
        r#"
        INSERT INTO ticket_learning_candidates (
            candidate_id, case_id, ticket_key, label, bundle_label, bundle_version,
            summary, proposed_actions_json, evidence_json, status, proposed_at,
            decided_at, decided_by, decision_notes, promoted_autonomy_level
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 'proposed', ?10, NULL, NULL, NULL, NULL)
        "#,
        params![
            candidate_id,
            case_id,
            case.ticket_key,
            case.label,
            case.bundle_label,
            case.bundle_version,
            summary.trim(),
            serde_json::to_string(&proposed_actions)?,
            serde_json::to_string(&evidence)?,
            now,
        ],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(case_id),
            actor_type: "learning_engine",
            action_type: "learning_candidate",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({
                "candidate_id": candidate_id,
                "summary": summary.trim(),
                "proposed_actions": proposed_actions,
            }),
        },
    )?;
    load_learning_candidate(root, &candidate_id)?
        .context("failed to load learning candidate after create")
}

pub(super) fn list_learning_candidates(
    root: &Path,
    label: Option<&str>,
    status: Option<&str>,
    limit: usize,
) -> Result<Vec<LearningCandidateView>> {
    let conn = open_ticket_db(root)?;
    let mut statement = conn.prepare(
        r#"
        SELECT candidate_id, case_id, ticket_key, label, bundle_label, bundle_version, summary,
               proposed_actions_json, evidence_json, status, proposed_at, decided_at, decided_by,
               decision_notes, promoted_autonomy_level
        FROM ticket_learning_candidates
        WHERE (?1 IS NULL OR label = ?1)
          AND (?2 IS NULL OR status = ?2)
        ORDER BY proposed_at DESC
        LIMIT ?3
        "#,
    )?;
    let rows = statement.query_map(
        params![label, status, limit as i64],
        map_learning_candidate_row,
    )?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub(super) fn load_learning_candidate(
    root: &Path,
    candidate_id: &str,
) -> Result<Option<LearningCandidateView>> {
    let conn = open_ticket_db(root)?;
    conn.query_row(
        r#"
        SELECT candidate_id, case_id, ticket_key, label, bundle_label, bundle_version, summary,
               proposed_actions_json, evidence_json, status, proposed_at, decided_at, decided_by,
               decision_notes, promoted_autonomy_level
        FROM ticket_learning_candidates
        WHERE candidate_id = ?1
        LIMIT 1
        "#,
        params![candidate_id],
        map_learning_candidate_row,
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub(super) fn decide_learning_candidate(
    root: &Path,
    candidate_id: &str,
    status: &str,
    decided_by: &str,
    notes: Option<&str>,
    promoted_autonomy_level: Option<&str>,
) -> Result<LearningCandidateView> {
    let mut conn = open_ticket_db(root)?;
    let candidate =
        load_learning_candidate(root, candidate_id)?.context("learning candidate not found")?;
    let canonical_status = canonical_learning_candidate_status(status)?;
    let promoted_autonomy_level = promoted_autonomy_level
        .map(canonical_autonomy_level)
        .transpose()?
        .map(ToOwned::to_owned);
    let now = now_iso_string();
    conn.execute(
        r#"
        UPDATE ticket_learning_candidates
        SET status = ?2,
            decided_at = ?3,
            decided_by = ?4,
            decision_notes = ?5,
            promoted_autonomy_level = ?6
        WHERE candidate_id = ?1
        "#,
        params![
            candidate_id,
            canonical_status,
            now,
            decided_by.trim(),
            notes.map(str::trim),
            promoted_autonomy_level,
        ],
    )?;
    record_audit(
        &mut conn,
        AuditRequest {
            ticket_key: &candidate.ticket_key,
            case_id: Some(&candidate.case_id),
            actor_type: "approver",
            action_type: "learning_candidate_decision",
            label: Some(&candidate.label),
            bundle_label: Some(&candidate.bundle_label),
            bundle_version: Some(candidate.bundle_version),
            details: json!({
                "candidate_id": candidate_id,
                "status": canonical_status,
                "decided_by": decided_by.trim(),
                "notes": notes.map(str::trim),
                "promoted_autonomy_level": promoted_autonomy_level,
            }),
        },
    )?;
    load_learning_candidate(root, candidate_id)?
        .context("failed to load learning candidate after decision")
}

pub(super) fn list_audit_records(
    root: &Path,
    ticket_key: Option<&str>,
    limit: usize,
) -> Result<Vec<TicketAuditRecord>> {
    let conn = open_ticket_db(root)?;
    let sql = if ticket_key.is_some() {
        r#"
        SELECT audit_id, ticket_key, case_id, actor_type, action_type, label, bundle_label,
               bundle_version, details_json, created_at
        FROM ticket_audit_log
        WHERE ticket_key = ?1
        ORDER BY created_at DESC
        LIMIT ?2
        "#
    } else {
        r#"
        SELECT audit_id, ticket_key, case_id, actor_type, action_type, label, bundle_label,
               bundle_version, details_json, created_at
        FROM ticket_audit_log
        ORDER BY created_at DESC
        LIMIT ?1
        "#
    };
    let mut statement = conn.prepare(sql)?;
    let rows = if let Some(ticket_key) = ticket_key {
        statement.query_map(params![ticket_key, limit as i64], map_audit_row)?
    } else {
        statement.query_map(params![limit as i64], map_audit_row)?
    };
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub(super) fn ensure_case_is_executable(case: &TicketCaseView) -> Result<()> {
    match case.state.as_str() {
        "executable" | "executing" => Ok(()),
        other => anyhow::bail!(
            "case {} is not executable; current state is {}",
            case.case_id,
            other
        ),
    }
}

pub(super) fn ensure_case_can_request_clarification(case: &TicketCaseView) -> Result<()> {
    match case.state.as_str() {
        "closed" | "done" | "completed" | "verified" | "writeback_pending" => anyhow::bail!(
            "case {} cannot request clarification from terminal/writeback state {}",
            case.case_id,
            case.state
        ),
        _ => Ok(()),
    }
}

pub(super) fn ensure_case_ready_for_writeback(case: &TicketCaseView) -> Result<()> {
    match case.state.as_str() {
        "writeback_pending" | "verifying" => Ok(()),
        other => anyhow::bail!(
            "case {} is not ready for writeback; current state is {}",
            case.case_id,
            other
        ),
    }
}

pub(super) fn canonical_clarification_target_type(raw: &str) -> Result<String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "requester" | "customer" | "ticket_requester" => Ok("requester".to_string()),
        "owner" | "founder" | "admin" => Ok("owner".to_string()),
        "internal" | "team" | "operator" => Ok("internal".to_string()),
        other => anyhow::bail!(
            "unsupported clarification target_type '{other}' (expected requester|owner|internal)"
        ),
    }
}

pub(super) fn canonical_clarification_target_channel(raw: &str) -> Result<String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "ticket" | "email" | "jami" | "tui" | "teams" | "whatsapp" => {
            Ok(raw.trim().to_ascii_lowercase())
        }
        other => anyhow::bail!(
            "unsupported clarification target_channel '{other}' (expected ticket|email|jami|tui|teams|whatsapp)"
        ),
    }
}

pub(super) fn canonical_clarification_resume_state(raw: &str) -> Result<String> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "planned" | "ready" | "executable" => Ok("executable".to_string()),
        other => {
            anyhow::bail!("unsupported clarification resume_state '{other}' (expected executable)")
        }
    }
}

pub(super) fn normalize_clarification_inputs(values: Vec<String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    values
        .into_iter()
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .filter(|value| seen.insert(value.to_ascii_lowercase()))
        .collect()
}

pub(super) fn record_failed_writeback(
    conn: &mut Connection,
    case: &TicketCaseView,
    operation: &str,
    payload: Value,
    error: &str,
) -> Result<()> {
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO ticket_writebacks (
            writeback_id, case_id, ticket_key, operation, payload_json, status, created_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, 'failed', ?6)
        "#,
        params![
            format!(
                "writeback-failed:{}:{}",
                case.case_id,
                stable_digest(&(operation.to_string() + error + now.as_str()))
            ),
            case.case_id,
            case.ticket_key,
            operation,
            serde_json::to_string(&json!({
                "payload": payload,
                "error": collapse_inline(error, 1000),
            }))?,
            now,
        ],
    )?;
    record_audit(
        conn,
        AuditRequest {
            ticket_key: &case.ticket_key,
            case_id: Some(&case.case_id),
            actor_type: "writeback_engine",
            action_type: "writeback_failed",
            label: Some(&case.label),
            bundle_label: Some(&case.bundle_label),
            bundle_version: Some(case.bundle_version),
            details: json!({
                "operation": operation,
                "error": collapse_inline(error, 1000),
            }),
        },
    )
}

pub(super) struct AuditRequest<'a> {
    pub(super) ticket_key: &'a str,
    pub(super) case_id: Option<&'a str>,
    pub(super) actor_type: &'a str,
    pub(super) action_type: &'a str,
    pub(super) label: Option<&'a str>,
    pub(super) bundle_label: Option<&'a str>,
    pub(super) bundle_version: Option<i64>,
    pub(super) details: Value,
}

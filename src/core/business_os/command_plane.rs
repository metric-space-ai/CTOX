// Origin: CTOX
// License: Apache-2.0

use super::app_runtime;
use super::control_command_types::ActiveExternalSqlControlCommand;
use super::policy::{BusinessOsPermission, BusinessOsScope, BusinessOsScopeType};
use super::session::{session_user_id, BusinessOsSession};
use super::store::{
    appsec_business_command_requires_data_write, authorize_recoverable_background_control_command,
    business_command_core_claim_with_authorization, command_inbound_channel,
    external_sql_command_in_flight_outcome, first_string_field, handle_app_lifecycle_command,
    handle_business_os_command, handle_mailserver_command, handle_module_command,
    handle_secret_command, handle_source_command, handle_workspace_control_command,
    is_appsec_business_command, is_ats_active_command, is_ats_mutating_command,
    is_customers_active_command, is_iot_active_command, is_outbound_active_command,
    is_recoverable_background_control_command_type, is_rxdb_control_command_type, now_ms,
    open_store, persist_business_command_lifecycle_projection,
    project_iot_business_command_outcome, pull_collection_record,
    record_business_module_lifecycle_event, record_command, record_report_command,
    recoverable_background_control_claim_authorization, run_channel_command,
    rxdb_authenticated_session, rxdb_command_session, stored_rxdb_business_command_outcome,
    upsert_rxdb_collection_record, write_rxdb_failed_control_command_outcome, BusinessCommand,
    BusinessOsReportMutation, ChannelCommandRequest, CommandOrigin, APPSEC_MODULE_ID,
};
use super::store_appsec_commands::handle_appsec_business_command;
use super::store_ats_commands::{handle_ats_active_command, handle_ats_mutating_command};
use super::store_customer_commands::handle_customers_active_command;
use super::store_office_commands::handle_office_control_command;
use super::store_outbound_commands::handle_outbound_active_command;
use super::store_policy::{enforce_command_policy, CommandPolicyRequirement};
use super::store_policy_audit::app_build_command_policy_target;
use super::store_projections::{
    is_business_chat_command, materialize_control_business_chat_state, upsert_business_record,
};
use crate::mission::channels;
use anyhow::Context;
use rusqlite::{params, OptionalExtension};
use serde_json::Value;
use std::path::Path;

pub(super) const EXACT_CONTROL_TYPES: [&str; 52] = [
    "ctox.app.access.grant",
    "ctox.app.access.revoke",
    "ctox.app.action.run",
    "ctox.app_store.install",
    "ctox.app_store.uninstall",
    "ctox.business_os.audit.list",
    "ctox.business_os.audit.retention",
    "ctox.business_os.audit.retention_policy.set",
    "ctox.business_os.backup.restore_drill",
    "ctox.business_os.branding.update",
    "ctox.business_os.support.export_diagnostics",
    "ctox.business_os.user.upsert",
    "ctox.business_os.why",
    "ctox.coding.turn",
    "ctox.command.cancel",
    "ctox.file.export",
    "ctox.file.materialize",
    "ctox.mailserver.delete_domain",
    "ctox.mailserver.delete_user",
    "ctox.mailserver.get_config",
    "ctox.mailserver.save_domain",
    "ctox.mailserver.save_user",
    "ctox.maintenance.client_ready",
    "ctox.module.assign_founder",
    "ctox.module.check_updates",
    "ctox.module.delete",
    "ctox.module.install_template",
    "ctox.module.list_versions",
    "ctox.module.release",
    "ctox.module.repair_lifecycle_projection",
    "ctox.module.rollback",
    "ctox.module.rollback_version",
    "ctox.module.save",
    "ctox.module.set_visible",
    "ctox.module.update",
    "ctox.office.settings.save",
    "ctox.runtime_settings.save",
    "ctox.secret.delete",
    "ctox.secret.list",
    "ctox.secret.put",
    "ctox.source.commit",
    "ctox.source.diff",
    "ctox.source.list_snapshots",
    "ctox.source.load",
    "ctox.source.log",
    "ctox.source.rollback_snapshot",
    "ctox.source.save",
    "ctox.subscription_auth.start",
    "ctox.task.delete",
    "ctox.task.update",
    "knowledge.command",
    "web_stack.person_research",
];

/// Accept a command that originated in trusted, in-process code (operator CLI,
/// server-side handlers, internal projections, tests). The claimed actor in
/// `client_context` is trusted. Network-originated commands MUST use
/// [`accept_rxdb_business_command_with_origin`] with [`CommandOrigin::ReplicatedPeer`].
pub fn accept_rxdb_business_command(root: &Path, document: Value) -> anyhow::Result<Value> {
    accept_rxdb_business_command_with_origin(root, document, CommandOrigin::TrustedLocal)
}

/// Accept a Business OS command, tagging it with its trust [`CommandOrigin`].
/// `ReplicatedPeer` commands (arriving over the WebRTC/RxDB data plane) cannot
/// authorize a privileged role from the browser-asserted actor; identity/role
/// for those comes only from a verified capability token.
pub fn accept_rxdb_business_command_with_origin(
    root: &Path,
    document: Value,
    origin: CommandOrigin,
) -> anyhow::Result<Value> {
    let command_id = document
        .get("command_id")
        .or_else(|| document.get("id"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .context("business command id is required")?
        .to_string();
    let command = BusinessCommand {
        origin,
        id: Some(command_id.clone()),
        module: document
            .get("module")
            .and_then(Value::as_str)
            .unwrap_or("ctox")
            .to_string(),
        command_type: document
            .get("command_type")
            .or_else(|| document.get("type"))
            .and_then(Value::as_str)
            .unwrap_or("business_os.command")
            .to_string(),
        record_id: document
            .get("record_id")
            .and_then(Value::as_str)
            .map(str::to_string),
        payload: document.get("payload").cloned().unwrap_or(Value::Null),
        client_context: document
            .get("client_context")
            .cloned()
            .unwrap_or(Value::Null),
    };
    let native_authorization = recoverable_background_control_claim_authorization(root, &command);
    let control_claim = if is_rxdb_control_command_type(&command.command_type) {
        Some(channels::claim_business_control_command(
            root,
            business_command_core_claim_with_authorization(
                &command_id,
                &command,
                native_authorization.as_ref(),
            )?,
        )?)
    } else {
        None
    };
    let _external_sql_execution_guard =
        if super::external_sql_sync::is_external_sql_command(&command.command_type) {
            match ActiveExternalSqlControlCommand::try_acquire(&command_id) {
                Some(guard) => Some(guard),
                None => return external_sql_command_in_flight_outcome(root, &command_id),
            }
        } else {
            None
        };
    let owns_new_control_claim = control_claim
        .as_ref()
        .is_some_and(|claim| claim.disposition == "new");
    let conn = open_store(root)?;
    let existing_status: Option<String> = conn
        .query_row(
            "SELECT status FROM business_commands WHERE command_id = ?1",
            params![command_id.as_str()],
            |row| row.get(0),
        )
        .optional()?;
    let resumes_recoverable_accepted_claim = control_claim
        .as_ref()
        .is_some_and(|claim| claim.disposition == "uncertain")
        && existing_status.as_deref() == Some("accepted")
        && is_recoverable_background_control_command_type(&command.command_type);
    if !owns_new_control_claim
        && !resumes_recoverable_accepted_claim
        && existing_status.as_deref() != Some("waiting_dependencies")
        && existing_status.is_some()
    {
        if let Some(stored_outcome) = stored_rxdb_business_command_outcome(&conn, &command_id)? {
            if let Ok(mut lifecycle_outcome) =
                channels::business_command_projection(root, &command_id)
            {
                if let Some(object) = lifecycle_outcome.as_object_mut() {
                    let stored_chat_id = stored_outcome
                        .get("chat_id")
                        .or_else(|| {
                            stored_outcome
                                .get("result")
                                .and_then(|result| result.get("chat_id"))
                        })
                        .cloned();
                    if let Some(chat_id) = stored_chat_id {
                        object.insert("chat_id".to_string(), chat_id);
                    }
                    for field in ["outbound_text", "response", "answer", "summary"] {
                        if !object.contains_key(field) {
                            if let Some(value) = stored_outcome.get(field) {
                                object.insert(field.to_string(), value.clone());
                            }
                        }
                    }
                    object.insert("ok".to_string(), Value::Bool(true));
                    object.insert("already_accepted".to_string(), Value::Bool(true));
                }
                persist_business_command_lifecycle_projection(root, &lifecycle_outcome)?;
                return Ok(lifecycle_outcome);
            }
            return Ok(stored_outcome);
        }
        return Ok(serde_json::json!({
            "id": command_id,
            "command_id": command_id,
            "status": "already_accepted"
        }));
    }
    drop(conn);
    let command = if resumes_recoverable_accepted_claim {
        match authorize_recoverable_background_control_command(root, &command) {
            Ok(command) => command,
            Err(error) => {
                return write_rxdb_failed_control_command_outcome(
                    root,
                    &command,
                    "recoverable_control_authorization",
                    error,
                );
            }
        }
    } else {
        command
    };
    if is_rxdb_control_command_type(&command.command_type) {
        let claim = control_claim.context("business control command claim is missing")?;
        match claim.disposition {
            "new" => {}
            "terminal" => {
                let terminal_status = claim.terminal_status.as_deref().unwrap_or("completed");
                return Ok(serde_json::json!({
                    "ok": terminal_status == "completed",
                    "id": command_id,
                    "command_id": command_id,
                    "status": terminal_status,
                    "execution_mode": "control",
                    "execution_task_id": "",
                    "target_task_id": "",
                    "target_record_id": command.record_id.clone().unwrap_or_default(),
                    "task_id": "",
                    "task_status": terminal_status,
                    "result": claim.result.unwrap_or(Value::Null),
                    "already_accepted": true,
                }));
            }
            _ if is_recoverable_background_control_command_type(&command.command_type) => {}
            _ => {
                if command.command_type == app_runtime::APP_ACTION_COMMAND_TYPE {
                    if let Some(snapshot) = app_runtime::admitted_snapshot(root, &command_id)? {
                        return Ok(serde_json::json!({
                            "ok": true,
                            "id": command_id,
                            "command_id": command_id,
                            "status": "accepted",
                            "execution_mode": "control",
                            "execution_phase": "accepted",
                            "terminal_status": "none",
                            "task_status": "accepted",
                            "_app_action_snapshot": snapshot,
                            "already_accepted": false,
                            "resumed": true,
                        }));
                    }
                } else {
                    return Ok(serde_json::json!({
                        "ok": false,
                        "id": command_id,
                        "command_id": command_id,
                        "status": "accepted",
                        "execution_mode": "control",
                        "execution_task_id": "",
                        "target_task_id": "",
                        "target_record_id": command.record_id.clone().unwrap_or_default(),
                        "task_id": "",
                        "task_status": "blocked",
                        "execution_phase": "blocked",
                        "terminal_status": "none",
                        "error_code": "dependency_missing",
                        "error_message": "control effect was durably claimed but has no terminal outcome; automatic replay is suppressed to prevent a duplicate side effect",
                        "retryable": false,
                        "already_accepted": true,
                    }));
                }
            }
        }
    }
    match command.command_type.as_str() {
        "ctox.maintenance.client_ready"
        | "ctox.task.update"
        | "ctox.task.delete"
        | "ctox.command.cancel"
        | "ctox.runtime_settings.save"
        | "ctox.office.settings.save"
        | "ctox.coding.turn"
        | "ctox.file.materialize"
        | "ctox.file.export" => {
            return handle_workspace_control_command(root, &command);
        }
        "ctox.app.action.run"
        | "ctox.app.access.grant"
        | "ctox.app.access.revoke"
        | "ctox.app_store.install"
        | "ctox.app_store.uninstall" => {
            return handle_app_lifecycle_command(root, &command_id, &command);
        }
        command_type if crate::coding_agents::is_coding_agent_command(command_type) => {
            let outcome = match rxdb_command_session(root, &command)
                .and_then(|_| crate::coding_agents::handle_business_command(root, &command))
            {
                Ok(outcome) => outcome,
                Err(error) => serde_json::json!({
                    "ok": false,
                    "provider": command
                        .payload
                        .get("provider")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown"),
                    "operation": command.command_type,
                    "stdout": "",
                    "stderr": error.to_string(),
                    "exit_code": 1
                }),
            };
            let status = if outcome.get("ok").and_then(Value::as_bool) == Some(false) {
                "failed"
            } else {
                "completed"
            };
            return write_rxdb_control_command_outcome(
                root,
                &command,
                status,
                None,
                Some(status),
                serde_json::json!({ "outcome": outcome }),
            );
        }
        "web_stack.person_research" => {
            if command.module.trim().is_empty() {
                return write_rxdb_failed_control_command_outcome(
                    root,
                    &command,
                    "person_research_authorization",
                    anyhow::anyhow!("web_stack.person_research requires a calling module"),
                );
            }
            return match enforce_command_policy(
                root,
                &command,
                |_| {
                    Ok(CommandPolicyRequirement::module(
                        BusinessOsPermission::DataRead,
                        &command.module,
                    ))
                },
                |_| super::person_research_command::start(root, command.clone()),
            ) {
                Ok(enforced) => enforced.into_outcome(),
                Err(error) => write_rxdb_failed_control_command_outcome(
                    root,
                    &command,
                    "person_research_authorization",
                    error,
                ),
            };
        }
        "knowledge.command" => {
            let args = command
                .payload
                .get("args")
                .and_then(Value::as_array)
                .context("knowledge.command payload.args array is required")?
                .iter()
                .map(|value| {
                    value
                        .as_str()
                        .map(str::to_string)
                        .context("knowledge.command args must be strings")
                })
                .collect::<anyhow::Result<Vec<_>>>()?;
            let outcome = crate::knowledge::dispatch_capturing(root, &args)?;
            return write_rxdb_control_command_outcome(
                root,
                &command,
                "completed",
                None,
                Some("completed"),
                outcome,
            );
        }
        "ctox.business_os.user.upsert"
        | "ctox.business_os.branding.update"
        | "ctox.business_os.audit.list"
        | "ctox.business_os.audit.retention"
        | "ctox.business_os.audit.retention_policy.set"
        | "ctox.business_os.backup.restore_drill"
        | "ctox.business_os.support.export_diagnostics"
        | "ctox.business_os.why" => {
            return handle_business_os_command(root, &command);
        }
        "ctox.secret.list"
        | "ctox.secret.put"
        | "ctox.secret.delete"
        | "ctox.subscription_auth.start" => {
            return handle_secret_command(root, &command);
        }
        "ctox.module.repair_lifecycle_projection"
        | "ctox.module.release"
        | "ctox.module.assign_founder"
        | "ctox.module.save"
        | "ctox.module.delete"
        | "ctox.module.install_template"
        | "ctox.module.update"
        | "ctox.module.set_visible"
        | "ctox.module.check_updates"
        | "ctox.module.rollback"
        | "ctox.module.list_versions"
        | "ctox.module.rollback_version" => {
            return handle_module_command(root, &command);
        }
        command_type
            if command_type.starts_with("office.document.")
                || command_type.starts_with("office.spreadsheet.") =>
        {
            let module_id = if command_type.starts_with("office.document.") {
                "documents"
            } else {
                "spreadsheets"
            };
            return enforce_command_policy(
                root,
                &command,
                |_| {
                    Ok(CommandPolicyRequirement::module(
                        BusinessOsPermission::DataWrite,
                        module_id,
                    ))
                },
                |_| match handle_office_control_command(root, &command) {
                    Ok(outcome) => write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        command.record_id.as_deref(),
                        Some("completed"),
                        outcome,
                    ),
                    Err(error) => {
                        let error_code = if error.to_string().contains("version_conflict") {
                            "version_conflict"
                        } else if error.to_string().contains("feature_dependency_pending") {
                            "feature_dependency_pending"
                        } else {
                            "office_engine_failed"
                        };
                        Err(write_failed_control_command_outcome_observably(
                            root,
                            &command,
                            command.record_id.as_deref(),
                            serde_json::json!({
                                "ok": false,
                                "error_code": error_code,
                                "error": error.to_string(),
                            }),
                            error,
                        ))
                    }
                },
            )?
            .into_outcome();
        }
        command_type if is_customers_active_command(command_type) => {
            return enforce_command_policy(
                root,
                &command,
                |_| {
                    Ok(CommandPolicyRequirement::module(
                        BusinessOsPermission::DataWrite,
                        "customers",
                    ))
                },
                |session| match handle_customers_active_command(root, session, &command) {
                    Ok(outcome) => write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        None,
                        Some("completed"),
                        outcome,
                    ),
                    Err(error) => Err(write_failed_control_command_outcome_observably(
                        root,
                        &command,
                        None,
                        serde_json::json!({
                            "ok": false,
                            "error": error.to_string(),
                        }),
                        error,
                    )),
                },
            )?
            .into_outcome();
        }
        command_type if super::external_sql_sync::is_external_sql_command(command_type) => {
            return enforce_command_policy(
                root,
                &command,
                |_| {
                    Ok(CommandPolicyRequirement::module(
                        super::external_sql_sync::data_write_permission(),
                        &command.module,
                    ))
                },
                |_| match super::external_sql_sync::handle_business_command(root, &command) {
                    Ok(outcome) => write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        command.record_id.as_deref(),
                        Some("completed"),
                        outcome,
                    ),
                    Err(error) => Err(write_failed_control_command_outcome_observably(
                        root,
                        &command,
                        command.record_id.as_deref(),
                        serde_json::json!({
                            "ok": false,
                            "error": error.to_string(),
                        }),
                        error,
                    )),
                },
            )?
            .into_outcome();
        }
        command_type if is_outbound_active_command(command_type) => {
            return enforce_command_policy(
                root,
                &command,
                |_| {
                    Ok(CommandPolicyRequirement::module(
                        BusinessOsPermission::DataWrite,
                        "outbound",
                    ))
                },
                |session| {
                    let outcome =
                        handle_outbound_active_command(root, session, &command_id, &command)?;
                    write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        command.record_id.as_deref(),
                        Some("completed"),
                        outcome,
                    )
                },
            )?
            .into_outcome();
        }
        command_type if is_iot_active_command(command_type) => {
            // §4A: the executor goes through the SAME iot::commands code path the
            // `ctox iot` CLI uses. This family still uses rxdb_command_session
            // to enforce a real chef/admin role via session_can_manage_all, so
            // an untrusted peer that falls through to the default `user` role is
            // rejected here with "chef or admin role required" instead of
            // slipping past the always-true `authenticated && !auth_required`
            // disjunct downstream.
            // Then write a completed/failed outcome whose `result.projections` the
            // rxdb_peer branch reprojects into the iot_* collections. Idempotent: a
            // replayed command short-circuits on the stored outcome above.
            let session = rxdb_command_session(root, &command)?;
            match crate::iot::commands::handle_business_command(
                root,
                command_type,
                &command.payload,
                &session,
            ) {
                Ok(outcome) => {
                    // Project engine state into the RxDB-visible business_records
                    // store via iot::projector (same code path as the rxdb_peer
                    // live stream). Failure to project must not silently drop the
                    // outcome, so surface it.
                    project_iot_business_command_outcome(root, &outcome)
                        .context("project iot business command outcome")?;
                    return write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        command.record_id.as_deref(),
                        Some("completed"),
                        outcome,
                    );
                }
                Err(error) => {
                    return Err(write_failed_control_command_outcome_observably(
                        root,
                        &command,
                        command.record_id.as_deref(),
                        serde_json::json!({
                            "ok": false,
                            "error": error.to_string(),
                        }),
                        error,
                    ));
                }
            }
        }
        command_type if is_ats_active_command(command_type) => {
            // Server-authoritative ATS gate checks: the browser asks the daemon
            // whether a deployment/presentation is allowed; the decision is
            // computed native-side from business_credentials/business_consents
            // via ats_gates and returned as the command outcome. Read-only, so
            // an authenticated peer suffices (no mutation, no projection).
            let session = rxdb_authenticated_session(root, &command)?;
            let outcome = handle_ats_active_command(root, &session, &command)?;
            return write_rxdb_control_command_outcome(
                root,
                &command,
                "completed",
                command.record_id.as_deref(),
                Some("completed"),
                outcome,
            );
        }
        command_type if is_ats_mutating_command(command_type) => {
            // Server-authoritative ATS mutations (write records, gated where
            // applicable). chef/admin role required, like the other active
            // mutating module families.
            let session = rxdb_command_session(root, &command)?;
            let outcome = handle_ats_mutating_command(root, &session, &command)?;
            return write_rxdb_control_command_outcome(
                root,
                &command,
                "completed",
                command.record_id.as_deref(),
                Some("completed"),
                outcome,
            );
        }
        command_type if command_type.starts_with("invoices.") => {
            // §5.6/§5.11 invoices module: server-authoritative accounting
            // mutations (draft CRUD, post, Storno/cancel, §17 credit notes).
            // An unknown invoices.* command, an auth failure, or a handler error
            // must persist a FAILED business_commands outcome AND propagate the
            // error — never fall through to generic queue acceptance. Reject an
            // unknown type before the session check so the error names the
            // unsupported type rather than an auth failure.
            let outcome = (|| -> anyhow::Result<Value> {
                anyhow::ensure!(
                    super::invoices::is_invoices_active_command(command_type),
                    "unsupported invoices command type: {command_type}"
                );
                let session = rxdb_command_session(root, &command)?;
                super::invoices::handle_invoices_active_command(root, &session, &command)
            })();
            match outcome {
                Ok(value) => {
                    return write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        command.record_id.as_deref(),
                        Some("completed"),
                        value,
                    );
                }
                Err(err) => {
                    return Err(write_failed_control_command_outcome_observably(
                        root,
                        &command,
                        command.record_id.as_deref(),
                        serde_json::json!({ "ok": false, "error": err.to_string() }),
                        err,
                    ));
                }
            }
        }
        command_type if command_type.starts_with("ctox.channel.") => {
            let mutation: ChannelCommandRequest = serde_json::from_value(command.payload.clone())
                .context("invalid ctox.channel payload")?;
            return enforce_command_policy(
                root,
                &command,
                |_| {
                    Ok(CommandPolicyRequirement::workspace(
                        BusinessOsPermission::IntegrationsManage,
                    ))
                },
                |session| {
                    let outcome = run_channel_command(root, session, command_type, mutation)?;
                    write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        None,
                        Some("completed"),
                        outcome,
                    )
                },
            )?
            .into_outcome();
        }
        command_type if is_appsec_business_command(command_type) => {
            let permission = if appsec_business_command_requires_data_write(command_type) {
                BusinessOsPermission::DataWrite
            } else {
                BusinessOsPermission::DataRead
            };
            return enforce_command_policy(
                root,
                &command,
                |_| {
                    Ok(CommandPolicyRequirement::module(
                        permission,
                        APPSEC_MODULE_ID,
                    ))
                },
                |session| match handle_appsec_business_command(root, session, &command) {
                    Ok(outcome) => {
                        let status = if outcome.get("ok").and_then(Value::as_bool) == Some(false) {
                            "failed"
                        } else {
                            "completed"
                        };
                        write_rxdb_control_command_outcome(
                            root,
                            &command,
                            status,
                            command.record_id.as_deref(),
                            Some(status),
                            outcome,
                        )
                    }
                    Err(error) => Err(write_failed_control_command_outcome_observably(
                        root,
                        &command,
                        command.record_id.as_deref(),
                        serde_json::json!({
                            "ok": false,
                            "error": error.to_string(),
                        }),
                        error,
                    )),
                },
            )?
            .into_outcome();
        }
        command_type if command_type.starts_with("ctox.ticket.") => {
            return enforce_command_policy(
                root,
                &command,
                |_| {
                    Ok(CommandPolicyRequirement::module(
                        BusinessOsPermission::SupportTriage,
                        "support",
                    ))
                },
                |_| {
                    let outcome = crate::mission::tickets::run_business_os_ticket_command(
                        root,
                        command_type,
                        &command.payload,
                    )?;
                    let task_id = outcome
                        .get("case_id")
                        .or_else(|| outcome.get("ticket_key"))
                        .and_then(Value::as_str)
                        .map(str::to_string);
                    write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        task_id.as_deref(),
                        Some("completed"),
                        outcome,
                    )
                },
            )?
            .into_outcome();
        }
        command_type if super::support::is_support_command(command_type) => {
            let permission = super::support::command_permission(command_type);
            return enforce_command_policy(
                root,
                &command,
                |_| Ok(CommandPolicyRequirement::module(permission, "support")),
                |session| match super::support::handle_business_command(root, session, &command) {
                    Ok(outcome) => write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        command.record_id.as_deref(),
                        Some("completed"),
                        outcome,
                    ),
                    Err(error) => Err(write_failed_control_command_outcome_observably(
                        root,
                        &command,
                        command.record_id.as_deref(),
                        serde_json::json!({
                            "ok": false,
                            "error": error.to_string(),
                        }),
                        error,
                    )),
                },
            )?
            .into_outcome();
        }
        command_type if super::threads::is_threads_command(command_type) => {
            let handle = |session: &BusinessOsSession| -> anyhow::Result<Value> {
                match super::threads::handle_business_command(root, session, &command) {
                    Ok(outcome) => write_rxdb_control_command_outcome(
                        root,
                        &command,
                        "completed",
                        None,
                        Some("completed"),
                        outcome,
                    ),
                    Err(error) => Err(write_failed_control_command_outcome_observably(
                        root,
                        &command,
                        None,
                        serde_json::json!({
                            "ok": false,
                            "error": error.to_string(),
                        }),
                        error,
                    )),
                }
            };
            if super::threads::requires_external_approval(command_type) {
                return enforce_command_policy(
                    root,
                    &command,
                    |session| {
                        let approval_id = command
                            .payload
                            .get("approval_request_id")
                            .or_else(|| command.payload.get("id"))
                            .and_then(Value::as_str)
                            .map(str::trim)
                            .filter(|value| !value.is_empty())
                            .or(command.record_id.as_deref())
                            .unwrap_or_default()
                            .to_owned();
                        let assigned_to_actor = if approval_id.is_empty() {
                            false
                        } else {
                            let actor_id = session_user_id(session).unwrap_or_default();
                            pull_collection_record(
                                root,
                                "ctox_task_approval_requests",
                                &approval_id,
                            )?
                            .and_then(|approval| {
                                first_string_field(&approval, &["reviewer_user_id"])
                            })
                            .map(|reviewer| reviewer == actor_id)
                            .unwrap_or(false)
                        };
                        Ok(CommandPolicyRequirement::scoped(
                            BusinessOsPermission::ExternalApprove,
                            BusinessOsScope {
                                scope_type: BusinessOsScopeType::Approval,
                                scope_id: if approval_id.is_empty() {
                                    None
                                } else {
                                    Some(approval_id)
                                },
                                assigned_to_actor,
                                owned_by_actor: false,
                            },
                        ))
                    },
                    handle,
                )?
                .into_outcome();
            }
            let session = rxdb_authenticated_session(root, &command)?;
            return handle(&session);
        }
        command_type if command_type.starts_with("ctox.report.") => {
            let mut mutation: BusinessOsReportMutation =
                serde_json::from_value(command.payload.clone())
                    .context("invalid ctox.report payload")?;
            if mutation.kind.trim().is_empty() {
                mutation.kind = command_type
                    .strip_prefix("ctox.report.")
                    .unwrap_or("bug")
                    .to_string();
            }
            mutation.client_context = command.client_context.clone();
            let session = rxdb_authenticated_session(root, &command)?;
            let accepted = record_report_command(
                root,
                &session,
                mutation,
                Some(command_id),
                command.record_id.clone(),
            )?;
            return Ok(serde_json::json!({
                "ok": true,
                "id": accepted.command_id,
                "command_id": accepted.command_id,
                "status": "accepted",
                "task_id": accepted.task_id.unwrap_or_default(),
                "task_status": accepted.task_status.unwrap_or_else(|| "accepted".to_string()),
                "report_id": accepted.report_id,
                "report_status": "open"
            }));
        }
        "ctox.source.load"
        | "ctox.source.save"
        | "ctox.source.list_snapshots"
        | "ctox.source.rollback_snapshot"
        | "ctox.source.commit"
        | "ctox.source.log"
        | "ctox.source.diff" => {
            return handle_source_command(root, &command);
        }
        "ctox.mailserver.get_config"
        | "ctox.mailserver.save_domain"
        | "ctox.mailserver.delete_domain"
        | "ctox.mailserver.save_user"
        | "ctox.mailserver.delete_user" => {
            return handle_mailserver_command(root, &command);
        }
        _ => {}
    }
    // CHOKEPOINT (DS-0.2 / H5+H9): every command type without a dedicated,
    // already-gated arm above falls through here into record_command, which
    // records it AND enqueues server work (create_ctox_queue_task) — previously
    // with no authorization. The exposure is the untrusted RxDB/WebRTC data
    // plane, so gate only ReplicatedPeer commands (TrustedLocal is the operator
    // CLI / in-process callers, already trusted, matching rxdb_session_from_
    // command). App-build commands carry their own AppsInstall/AppsModify gate
    // inside record_command; every other fall-through is a record-mutating data
    // command (source.parse, matching.*, business_os.chat.task / cv-print,
    // documents.*), so require module-scoped DataWrite. The session is the
    // capability-token actor, so an unprivileged / unauthenticated replicated
    // peer (inert "user", no grant) is denied and never reaches record_command
    // or create_ctox_queue_task.
    if matches!(command.origin, CommandOrigin::ReplicatedPeer)
        && app_build_command_policy_target(&command).is_none()
    {
        return enforce_command_policy(
            root,
            &command,
            |_| {
                let permission = if command.command_type == "business_os.context.ask" {
                    BusinessOsPermission::DataRead
                } else {
                    BusinessOsPermission::DataWrite
                };
                Ok(CommandPolicyRequirement::module(
                    permission,
                    &command.module,
                ))
            },
            |_| {
                let accepted = record_command(root, command.clone())?;
                Ok(serde_json::to_value(accepted)?)
            },
        )?
        .into_outcome();
    }
    let accepted = record_command(root, command)?;
    Ok(serde_json::to_value(accepted)?)
}

pub(super) fn write_rxdb_control_command_outcome(
    root: &Path,
    command: &BusinessCommand,
    status: &str,
    task_id: Option<&str>,
    task_status: Option<&str>,
    result: Value,
) -> anyhow::Result<Value> {
    write_rxdb_control_command_state(root, command, status, task_id, task_status, result, true)
}

fn write_failed_control_command_outcome_observably(
    root: &Path,
    command: &BusinessCommand,
    task_id: Option<&str>,
    result: Value,
    command_error: anyhow::Error,
) -> anyhow::Error {
    match write_rxdb_control_command_outcome(
        root,
        command,
        "failed",
        task_id,
        Some("failed"),
        result,
    ) {
        Ok(_) => command_error,
        Err(outcome_error) => {
            let command_id = command.id.as_deref().unwrap_or("<missing>");
            let command_type = command.command_type.as_str();
            eprintln!(
                "[business-os] failed to write failed control command outcome: \
                 command_id={command_id} command_type={command_type} \
                 outcome_error={outcome_error:#}; command_error={command_error:#}"
            );
            command_error.context(format!(
                "failed to write failed control command outcome for \
                 command_id={command_id} command_type={command_type}: {outcome_error:#}"
            ))
        }
    }
}

pub(super) fn write_rxdb_control_command_progress(
    root: &Path,
    command: &BusinessCommand,
    status: &str,
    result: Value,
) -> anyhow::Result<Value> {
    write_rxdb_control_command_state(root, command, status, None, Some(status), result, false)
}

fn write_rxdb_control_command_state(
    root: &Path,
    command: &BusinessCommand,
    status: &str,
    task_id: Option<&str>,
    task_status: Option<&str>,
    mut result: Value,
    terminal: bool,
) -> anyhow::Result<Value> {
    let command_id = command.id.as_deref().context("command id is required")?;
    let now = now_ms() as i64;
    let target_task_id = if command.command_type.starts_with("ctox.task.") {
        task_id.unwrap_or_default()
    } else {
        ""
    };
    let target_record_id = command
        .record_id
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .or_else(|| {
            (!command.command_type.starts_with("ctox.task."))
                .then_some(task_id)
                .flatten()
        })
        .unwrap_or_default();
    if let Some(object) = result.as_object_mut() {
        object
            .entry("status".to_string())
            .or_insert_with(|| Value::String(status.to_string()));
        object.insert(
            "task_status".to_string(),
            Value::String(task_status.unwrap_or(status).to_string()),
        );
    }
    if !terminal && is_rxdb_control_command_type(&command.command_type) {
        channels::progress_business_control_command(root, command_id, status, &result)?;
    }
    if terminal && is_rxdb_control_command_type(&command.command_type) {
        let terminal_status = match status {
            "completed" => "completed",
            "cancelled" => "cancelled",
            _ => "failed",
        };
        channels::complete_business_control_command(
            root,
            command_id,
            terminal_status,
            &result,
            (terminal_status == "failed")
                .then(|| {
                    result
                        .get("error")
                        .or_else(|| result.pointer("/outcome/stderr"))
                        .and_then(Value::as_str)
                })
                .flatten(),
        )?;
    }
    let conn = open_store(root)?;
    conn.execute(
        "INSERT INTO business_commands
            (command_id, module, command_type, record_id, status, payload_json, client_context_json, observed_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)
         ON CONFLICT(command_id) DO UPDATE SET
            module = excluded.module,
            command_type = excluded.command_type,
            record_id = excluded.record_id,
            status = excluded.status,
            payload_json = excluded.payload_json,
            client_context_json = excluded.client_context_json,
            observed_at_ms = excluded.observed_at_ms",
        params![
            command_id,
            command.module,
            command.command_type,
            command.record_id.clone().unwrap_or_default(),
            status,
            serde_json::to_string(&command.payload)?,
            serde_json::to_string(&command.client_context)?,
            now
        ],
    )?;
    let chat_id = if is_business_chat_command(command) {
        Some(materialize_control_business_chat_state(
            root, &conn, command_id, command, status, &result, terminal, now,
        )?)
    } else {
        None
    };
    let mut projection = serde_json::json!({
        "id": command_id,
        "command_id": command_id,
        "module": command.module.clone(),
        "command_type": command.command_type.clone(),
        "record_id": command.record_id.clone().unwrap_or_default(),
        "status": status,
        "execution_mode": "control",
        "execution_task_id": "",
        "target_task_id": target_task_id,
        "target_record_id": target_record_id,
        "inbound_channel": command_inbound_channel(command),
        "task_id": task_id.unwrap_or_default(),
        "task_status": task_status.unwrap_or(status),
        "payload": command.payload.clone(),
        "client_context": command.client_context.clone(),
        "result": result.clone(),
        "error_code": result.get("error_code").cloned().unwrap_or(Value::Null),
        "error_message": result.get("error").cloned().unwrap_or(Value::Null),
        "updated_at_ms": now
    });
    if let Some(chat_id) = chat_id.as_deref() {
        projection["chat_id"] = Value::String(chat_id.to_string());
    }
    upsert_business_record(
        &conn,
        "business_commands",
        command_id,
        now,
        projection.clone(),
    )?;
    record_business_module_lifecycle_event(root, command, status, &result)?;
    upsert_rxdb_collection_record(root, "business_commands", command_id, now, projection)?;
    Ok(serde_json::json!({
        "ok": true,
        "id": command_id,
        "command_id": command_id,
        "status": status,
        "execution_mode": "control",
        "execution_task_id": "",
        "target_task_id": target_task_id,
        "target_record_id": target_record_id,
        "task_id": task_id.unwrap_or_default(),
        "task_status": task_status.unwrap_or(status),
        "error_code": result.get("error_code").cloned().unwrap_or(Value::Null),
        "error_message": result.get("error").cloned().unwrap_or(Value::Null),
        "chat_id": chat_id,
        "result": result
    }))
}

#[cfg(test)]
mod tests {
    use super::super::store::{business_command_core_claim, rxdb_store_path};
    use super::super::store_projections::tests::create_repair_rxdb_tables;
    use super::*;
    use rusqlite::Connection;
    use std::collections::BTreeSet;
    use tempfile::tempdir;

    fn rust_brace_depth_after(mut depth: usize, line: &str) -> usize {
        let mut quoted = false;
        let mut escaped = false;
        let mut chars = line.chars().peekable();
        while let Some(character) = chars.next() {
            if quoted {
                if escaped {
                    escaped = false;
                } else if character == '\\' {
                    escaped = true;
                } else if character == '"' {
                    quoted = false;
                }
                continue;
            }
            if character == '/' && chars.peek() == Some(&'/') {
                break;
            }
            match character {
                '"' => quoted = true,
                '{' => depth += 1,
                '}' => depth = depth.checked_sub(1).expect("unbalanced Rust braces"),
                _ => {}
            }
        }
        depth
    }

    fn dispatcher_exact_control_types(source: &str) -> BTreeSet<String> {
        let function_start = source
            .find("pub fn accept_rxdb_business_command_with_origin")
            .expect("accept_rxdb_business_command_with_origin must exist");
        let match_start = source[function_start..]
            .find("match command.command_type.as_str()")
            .map(|offset| function_start + offset)
            .expect("command dispatcher match must exist");
        let fallback = source[match_start..]
            .find("        _ => {}")
            .map(|offset| match_start + offset)
            .expect("command dispatcher fallback must exist");
        let classifier = &source[match_start..fallback];
        let mut lines = classifier.lines();
        let match_line = lines
            .next()
            .expect("command dispatcher match must have a body");
        let mut brace_depth = rust_brace_depth_after(0, match_line);
        let mut exact_types = BTreeSet::new();
        let mut arm_pattern = String::new();

        for line in lines {
            let trimmed = line.trim();
            if brace_depth == 1 && (trimmed.starts_with('"') || !arm_pattern.is_empty()) {
                if !arm_pattern.is_empty() {
                    arm_pattern.push(' ');
                }
                arm_pattern.push_str(trimmed);
                if arm_pattern.contains("=>") {
                    let pattern = arm_pattern
                        .split_once("=>")
                        .expect("literal dispatcher arm must contain =>")
                        .0;
                    exact_types.extend(pattern.split('"').skip(1).step_by(2).map(str::to_string));
                    arm_pattern.clear();
                }
            }
            brace_depth = rust_brace_depth_after(brace_depth, line);
        }

        assert!(
            arm_pattern.is_empty(),
            "unterminated string-literal dispatcher arm: {arm_pattern}"
        );
        exact_types
    }

    #[test]
    fn business_command_inventory_matches_exact_control_types() {
        let dispatcher_types = dispatcher_exact_control_types(include_str!("command_plane.rs"));
        let declared_types = EXACT_CONTROL_TYPES
            .into_iter()
            .map(str::to_string)
            .collect::<BTreeSet<_>>();
        assert_eq!(
            declared_types.len(),
            EXACT_CONTROL_TYPES.len(),
            "EXACT_CONTROL_TYPES contains duplicates"
        );

        let dispatcher_only = dispatcher_types
            .difference(&declared_types)
            .cloned()
            .collect::<Vec<_>>();
        let constant_only = declared_types
            .difference(&dispatcher_types)
            .cloned()
            .collect::<Vec<_>>();
        assert!(
            dispatcher_only.is_empty() && constant_only.is_empty(),
            "EXACT_CONTROL_TYPES and the dispatcher's string-literal arms differ; \
             dispatcher_only={dispatcher_only:?}, constant_only={constant_only:?}"
        );
    }

    #[test]
    fn control_command_outcome_updates_outbox_intake_projection() -> anyhow::Result<()> {
        let root = tempdir()?;
        let command_id = "cmd_appsec_outbox_race";
        drop(create_repair_rxdb_tables(root.path())?);
        let command = BusinessCommand {
            origin: CommandOrigin::TrustedLocal,
            id: Some(command_id.to_string()),
            module: APPSEC_MODULE_ID.to_string(),
            command_type: "ctox.appsec.tools.doctor".to_string(),
            record_id: Some("runtime/appsec/test".to_string()),
            payload: serde_json::json!({ "profile": "full" }),
            client_context: serde_json::json!({ "actor": { "id": "local-dev" } }),
        };

        let claim = channels::claim_business_control_command(
            root.path(),
            business_command_core_claim(command_id, &command)?,
        )?;
        assert_eq!(claim.disposition, "new");

        let conn = open_store(root.path())?;
        conn.execute(
            "INSERT INTO business_commands
                (command_id, module, command_type, record_id, status, payload_json, client_context_json, observed_at_ms)
             VALUES (?1, ?2, ?3, ?4, 'accepted', ?5, ?6, 1)",
            params![
                command_id,
                command.module,
                command.command_type,
                command.record_id,
                serde_json::to_string(&command.payload)?,
                serde_json::to_string(&command.client_context)?,
            ],
        )?;
        drop(conn);

        write_rxdb_control_command_progress(
            root.path(),
            &command,
            "running",
            serde_json::json!({ "ok": true, "status": "running" }),
        )?;

        let outcome = write_rxdb_control_command_outcome(
            root.path(),
            &command,
            "completed",
            command.record_id.as_deref(),
            Some("completed"),
            serde_json::json!({ "ok": true }),
        )?;
        assert_eq!(
            outcome.get("status").and_then(Value::as_str),
            Some("completed")
        );

        let conn = open_store(root.path())?;
        let (count, status): (i64, String) = conn.query_row(
            "SELECT COUNT(*), MAX(status) FROM business_commands WHERE command_id = ?1",
            params![command_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(count, 1);
        assert_eq!(status, "completed");
        let rxdb_conn = Connection::open(rxdb_store_path(root.path()))?;
        let projected: String = rxdb_conn.query_row(
            "SELECT data FROM ctox_business_os__business_commands__v1 WHERE id = ?1",
            params![command_id],
            |row| row.get(0),
        )?;
        let projected: Value = serde_json::from_str(&projected)?;
        assert_eq!(projected["status"], "completed");
        assert_eq!(projected["task_status"], "completed");
        assert_eq!(projected["result"]["status"], "completed");
        assert_eq!(projected["result"]["task_status"], "completed");
        Ok(())
    }

    #[test]
    fn failed_outcome_write_is_observable() {
        let command = BusinessCommand {
            origin: CommandOrigin::TrustedLocal,
            id: None,
            module: "invoices".to_string(),
            command_type: "invoices.invoice.create".to_string(),
            record_id: Some("invoice-observability-test".to_string()),
            payload: serde_json::json!({}),
            client_context: serde_json::json!({}),
        };

        let error = write_failed_control_command_outcome_observably(
            Path::new("unused-because-the-command-id-is-missing"),
            &command,
            command.record_id.as_deref(),
            serde_json::json!({ "ok": false, "error": "handler failed after mutation" }),
            anyhow::anyhow!("handler failed after mutation"),
        );
        let observable = format!("{error:#}");
        assert!(observable.contains("failed to write failed control command outcome"));
        assert!(observable.contains("command_id=<missing>"));
        assert!(observable.contains("command_type=invoices.invoice.create"));
        assert!(observable.contains("command id is required"));
        assert!(observable.contains("handler failed after mutation"));

        let source = include_str!("command_plane.rs");
        let ignored_write = ["let _ = ", "write_rxdb_control_command_outcome("].concat();
        assert!(
            !source.contains(&ignored_write),
            "failed outcome writes must reach the observable helper"
        );
    }

    #[test]
    fn control_command_outcome_updates_an_existing_intake_projection() -> anyhow::Result<()> {
        let temp = tempdir()?;
        let root = temp.path();
        let command_id = "cmd_long_running_control_outcome";
        let command = BusinessCommand {
            origin: CommandOrigin::TrustedLocal,
            id: Some(command_id.to_string()),
            module: "inventory".to_string(),
            command_type: "external_sql.sync.refresh".to_string(),
            record_id: None,
            payload: serde_json::json!({ "input": { "mode": "full" } }),
            client_context: serde_json::json!({ "actor": { "id": "mcp-admin" } }),
        };

        let claim = channels::claim_business_control_command(
            root,
            business_command_core_claim(command_id, &command)?,
        )?;
        assert_eq!(claim.disposition, "new");

        let conn = open_store(root)?;
        conn.execute(
            "INSERT INTO business_commands
                (command_id, module, command_type, record_id, status, payload_json, client_context_json, observed_at_ms)
             VALUES (?1, 'inventory', 'external_sql.sync.refresh', '', 'accepted', '{}', '{}', 1)",
            params![command_id],
        )?;
        drop(conn);

        let outcome = write_rxdb_control_command_outcome(
            root,
            &command,
            "completed",
            None,
            Some("completed"),
            serde_json::json!({ "ok": true, "synced": 1 }),
        )?;
        assert_eq!(
            outcome.get("status").and_then(Value::as_str),
            Some("completed")
        );

        let conn = open_store(root)?;
        let (count, status): (i64, String) = conn.query_row(
            "SELECT COUNT(*), MAX(status) FROM business_commands WHERE command_id = ?1",
            params![command_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(count, 1);
        assert_eq!(status, "completed");
        Ok(())
    }
}

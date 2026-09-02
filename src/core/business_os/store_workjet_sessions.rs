// Origin: CTOX
// License: AGPL-3.0-only

//! Native-owned Workjet session register command handlers.
//!
//! `workjet_sessions` is the server-authoritative binding between a logical
//! Workjet session and its one active working copy. Transfer lifecycle writes
//! deliberately do not live in this package; `workjet_session_transfers` is
//! registered and projected here only as the Package-1 journal contract.

use super::store::{
    open_store, outbound_load_record, outbound_load_records_by_string_field,
    upsert_business_record, upsert_rxdb_collection_record, BusinessCommand,
};
use anyhow::Context;
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::path::Path;

pub(super) const SESSIONS_COLLECTION: &str = "workjet_sessions";
pub(super) const TRANSFERS_COLLECTION: &str = "workjet_session_transfers";
const PROJECTS_COLLECTION: &str = "workjet_projects";
const WORKING_COPIES_COLLECTION: &str = "workjet_working_copies";

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct WorkjetSessionTransferTransition {
    pub(super) from: &'static str,
    pub(super) verb: &'static str,
    pub(super) to: &'static str,
}

/// The complete RFC §12 transfer state machine. Keeping verbs on every edge
/// makes illegal state changes rejectable without spreading lifecycle rules
/// across individual handlers.
pub(super) const WORKJET_SESSION_TRANSFER_STATES: [&str; 12] = [
    "pause_requested",
    "packing",
    "packed",
    "shipping",
    "applying",
    "applied",
    "switching",
    "resuming",
    "completed",
    "aborting",
    "rolled_back",
    "failed",
];

pub(super) const WORKJET_SESSION_TRANSFER_TRANSITIONS: [WorkjetSessionTransferTransition; 27] = [
    WorkjetSessionTransferTransition {
        from: "pause_requested",
        verb: "pause_ack",
        to: "packing",
    },
    WorkjetSessionTransferTransition {
        from: "pause_requested",
        verb: "abort",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "pause_requested",
        verb: "timeout",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "packing",
        verb: "pack_complete",
        to: "packed",
    },
    WorkjetSessionTransferTransition {
        from: "packing",
        verb: "abort",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "packing",
        verb: "timeout",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "packed",
        verb: "shipping_start",
        to: "shipping",
    },
    WorkjetSessionTransferTransition {
        from: "packed",
        verb: "abort",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "packed",
        verb: "timeout",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "shipping",
        verb: "apply_start",
        to: "applying",
    },
    WorkjetSessionTransferTransition {
        from: "shipping",
        verb: "abort",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "shipping",
        verb: "timeout",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "applying",
        verb: "apply_complete",
        to: "applied",
    },
    WorkjetSessionTransferTransition {
        from: "applying",
        verb: "abort",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "applying",
        verb: "timeout",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "applied",
        verb: "confirm_working_copy",
        to: "switching",
    },
    WorkjetSessionTransferTransition {
        from: "applied",
        verb: "abort",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "applied",
        verb: "timeout",
        to: "aborting",
    },
    WorkjetSessionTransferTransition {
        from: "switching",
        verb: "commit_complete",
        to: "resuming",
    },
    WorkjetSessionTransferTransition {
        from: "switching",
        verb: "abort",
        to: "failed",
    },
    WorkjetSessionTransferTransition {
        from: "switching",
        verb: "timeout",
        to: "failed",
    },
    WorkjetSessionTransferTransition {
        from: "switching",
        verb: "commit_failed",
        to: "failed",
    },
    WorkjetSessionTransferTransition {
        from: "resuming",
        verb: "resume_ack",
        to: "completed",
    },
    WorkjetSessionTransferTransition {
        from: "resuming",
        verb: "abort",
        to: "failed",
    },
    WorkjetSessionTransferTransition {
        from: "resuming",
        verb: "timeout",
        to: "failed",
    },
    WorkjetSessionTransferTransition {
        from: "aborting",
        verb: "compensation_complete",
        to: "rolled_back",
    },
    WorkjetSessionTransferTransition {
        from: "aborting",
        verb: "compensation_failed",
        to: "failed",
    },
];

pub(super) fn workjet_session_transfer_next_state(state: &str, verb: &str) -> Option<&'static str> {
    WORKJET_SESSION_TRANSFER_TRANSITIONS
        .iter()
        .find(|transition| transition.from == state && transition.verb == verb)
        .map(|transition| transition.to)
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionCreatePayload {
    #[serde(default, rename = "inbound_channel")]
    _inbound_channel: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
    project_id: String,
    working_copy_id: String,
    #[serde(default)]
    thread_id: Option<String>,
    #[serde(default)]
    coding_session_id: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionListPayload {
    #[serde(default, rename = "inbound_channel")]
    _inbound_channel: Option<String>,
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct SessionDeletePayload {
    #[serde(default, rename = "inbound_channel")]
    _inbound_channel: Option<String>,
    #[serde(default)]
    session_id: Option<String>,
}

pub(super) fn handle_workjet_session_store_command(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
    authorized_owner_email: Option<&str>,
    can_manage_all_records: bool,
) -> anyhow::Result<Value> {
    migrate_signed_owner_alias(root, authorized_owner_user_id, authorized_owner_email)?;
    match command.command_type.as_str() {
        "ctox.workjet.session.create" => {
            handle_workjet_session_create_command(root, command, authorized_owner_user_id)
        }
        "ctox.workjet.session.list" => {
            handle_workjet_session_list_command(root, command, authorized_owner_user_id)
        }
        "ctox.workjet.session.delete" => handle_workjet_session_delete_command(
            root,
            command,
            authorized_owner_user_id,
            can_manage_all_records,
        ),
        other => anyhow::bail!("unsupported Workjet session command type: {other}"),
    }
}

pub(super) fn migrate_signed_owner_alias(
    root: &Path,
    authorized_owner_user_id: &str,
    authorized_owner_email: Option<&str>,
) -> anyhow::Result<()> {
    let owner_user_id = bounded_required(authorized_owner_user_id, "owner_user_id", 256)?;
    let Some(owner_email) = authorized_owner_email else {
        return Ok(());
    };
    let owner_email = bounded_required(owner_email, "owner_email", 320)?.to_ascii_lowercase();
    anyhow::ensure!(
        owner_email.contains('@'),
        "owner_email must be an email address"
    );
    if owner_email == owner_user_id.to_ascii_lowercase() {
        return Ok(());
    }

    let conn = open_store(root)?;
    let now = super::store::now_ms() as i64;
    for collection in [
        PROJECTS_COLLECTION,
        WORKING_COPIES_COLLECTION,
        SESSIONS_COLLECTION,
        TRANSFERS_COLLECTION,
    ] {
        let records = outbound_load_records_by_string_field(
            &conn,
            collection,
            "owner_user_id",
            &owner_email,
        )?;
        for mut record in records {
            if record.get("is_deleted").and_then(Value::as_bool) == Some(true) {
                continue;
            }
            let record_id = record
                .get("id")
                .and_then(Value::as_str)
                .with_context(|| format!("{collection} owner migration record has no id"))?
                .to_owned();
            record["owner_user_id"] = Value::String(owner_user_id.clone());
            record["updated_at_ms"] = Value::from(now);
            persist_and_project_idempotently(root, &conn, collection, &record_id, now, record)?;
        }
    }
    Ok(())
}

pub(super) fn workjet_session_record_policy_scope(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
    authorized_owner_email: Option<&str>,
) -> anyhow::Result<super::policy::BusinessOsScope> {
    migrate_signed_owner_alias(root, authorized_owner_user_id, authorized_owner_email)?;
    let owner_user_id = bounded_required(authorized_owner_user_id, "owner_user_id", 256)?;
    let conn = open_store(root)?;
    let (scope_id, owned_by_actor) = match command.command_type.as_str() {
        "ctox.workjet.session.create" => {
            let payload = parse_create_payload(command)?;
            let project_id = bounded_required(&payload.project_id, "project_id", 128)?;
            let working_copy_id =
                bounded_required(&payload.working_copy_id, "working_copy_id", 160)?;
            let session_id =
                deterministic_session_id(&project_id, &working_copy_id, &owner_user_id);
            ensure_asserted_session_id(command, payload.session_id.as_deref(), &session_id)?;
            let project = outbound_load_record(&conn, PROJECTS_COLLECTION, &project_id)?;
            let working_copy =
                outbound_load_record(&conn, WORKING_COPIES_COLLECTION, &working_copy_id)?;
            let owned = project.as_ref().is_some_and(|record| {
                record.get("owner_user_id").and_then(Value::as_str) == Some(owner_user_id.as_str())
            }) && working_copy.as_ref().is_some_and(|record| {
                record.get("owner_user_id").and_then(Value::as_str) == Some(owner_user_id.as_str())
                    && record.get("project_id").and_then(Value::as_str) == Some(project_id.as_str())
            });
            (session_id, owned)
        }
        "ctox.workjet.session.delete" => {
            let session_id = session_id_for_delete(command)?;
            let owned = outbound_load_record(&conn, SESSIONS_COLLECTION, &session_id)?
                .as_ref()
                .is_some_and(|record| {
                    record.get("owner_user_id").and_then(Value::as_str)
                        == Some(owner_user_id.as_str())
                });
            (session_id, owned)
        }
        other => anyhow::bail!("{other} does not use a record-scoped DataWrite decision"),
    };
    Ok(super::policy::BusinessOsScope {
        scope_type: super::policy::BusinessOsScopeType::Record,
        scope_id: Some(format!("{SESSIONS_COLLECTION}/{scope_id}")),
        assigned_to_actor: false,
        owned_by_actor,
    })
}

fn handle_workjet_session_create_command(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
) -> anyhow::Result<Value> {
    let payload = parse_create_payload(command)?;
    let owner_user_id = bounded_required(authorized_owner_user_id, "owner_user_id", 256)?;
    let project_id = bounded_required(&payload.project_id, "project_id", 128)?;
    let working_copy_id = bounded_required(&payload.working_copy_id, "working_copy_id", 160)?;
    let thread_id = optional_bounded(payload.thread_id, "thread_id", 160)?;
    let coding_session_id = optional_bounded(payload.coding_session_id, "coding_session_id", 128)?;
    let session_id = deterministic_session_id(&project_id, &working_copy_id, &owner_user_id);
    ensure_asserted_session_id(command, payload.session_id.as_deref(), &session_id)?;

    let conn = open_store(root)?;
    let project = outbound_load_record(&conn, PROJECTS_COLLECTION, &project_id)?
        .context("Workjet project not found")?;
    ensure_owned(Some(&project), &owner_user_id, "project", false)?;
    anyhow::ensure!(
        project.get("status").and_then(Value::as_str) != Some("archived"),
        "cannot create a session for an archived project"
    );
    let working_copy = outbound_load_record(&conn, WORKING_COPIES_COLLECTION, &working_copy_id)?
        .context("Workjet working copy not found")?;
    ensure_owned(Some(&working_copy), &owner_user_id, "working copy", false)?;
    anyhow::ensure!(
        working_copy.get("project_id").and_then(Value::as_str) == Some(project_id.as_str()),
        "working copy belongs to a different project"
    );
    anyhow::ensure!(
        working_copy.get("status").and_then(Value::as_str) == Some("active"),
        "Workjet session requires an active working copy"
    );
    let computer_id = bounded_required(
        working_copy
            .get("computer_id")
            .and_then(Value::as_str)
            .context("working copy has no computer_id")?,
        "computer_id",
        256,
    )?;

    for other in outbound_load_records_by_string_field(
        &conn,
        SESSIONS_COLLECTION,
        "working_copy_id",
        &working_copy_id,
    )? {
        if other.get("id").and_then(Value::as_str) == Some(session_id.as_str())
            || other.get("is_deleted").and_then(Value::as_bool) == Some(true)
        {
            continue;
        }
        anyhow::ensure!(
            !matches!(
                other.get("run_status").and_then(Value::as_str),
                Some("running" | "resuming")
            ),
            "working copy already has an authoritative running session"
        );
    }

    let existing = outbound_load_record(&conn, SESSIONS_COLLECTION, &session_id)?;
    ensure_owned(existing.as_ref(), &owner_user_id, "session", false)?;
    let now = super::store::now_ms() as i64;
    let created_at_ms = existing
        .as_ref()
        .and_then(|record| record.get("created_at_ms"))
        .and_then(Value::as_i64)
        .unwrap_or(now);
    let updated_at_ms = existing
        .as_ref()
        .and_then(|record| record.get("updated_at_ms"))
        .and_then(Value::as_i64)
        .unwrap_or(now);
    let mut session = serde_json::json!({
        "id": session_id,
        "project_id": project_id,
        "working_copy_id": working_copy_id,
        "computer_id": computer_id,
        "run_status": "running",
        "fence_epoch": 0,
        "owner_user_id": owner_user_id,
        "created_at_ms": created_at_ms,
        "updated_at_ms": updated_at_ms,
        "is_deleted": false,
    });
    if let Some(thread_id) = thread_id {
        session["thread_id"] = Value::String(thread_id);
    }
    if let Some(coding_session_id) = coding_session_id {
        session["coding_session_id"] = Value::String(coding_session_id);
    }

    if let Some(existing) = existing.as_ref() {
        anyhow::ensure!(
            stable_record_content(existing) == stable_record_content(&session),
            "Workjet session already exists with different content"
        );
    }
    let session = persist_and_project_idempotently(
        root,
        &conn,
        SESSIONS_COLLECTION,
        &session_id,
        now,
        session,
    )?;
    Ok(serde_json::json!({
        "ok": true,
        "collection": SESSIONS_COLLECTION,
        "session": session,
    }))
}

fn handle_workjet_session_list_command(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
) -> anyhow::Result<Value> {
    let payload: SessionListPayload = serde_json::from_value(command.payload.clone())
        .context("invalid ctox.workjet.session.list payload")?;
    let owner_user_id = bounded_required(authorized_owner_user_id, "owner_user_id", 256)?;
    let limit = payload.limit.unwrap_or(100).clamp(1, 100);
    let conn = open_store(root)?;
    let mut sessions = outbound_load_records_by_string_field(
        &conn,
        SESSIONS_COLLECTION,
        "owner_user_id",
        &owner_user_id,
    )?;
    sessions.retain(|session| session.get("is_deleted").and_then(Value::as_bool) != Some(true));
    sessions.sort_by(|left, right| {
        right
            .get("updated_at_ms")
            .and_then(Value::as_i64)
            .cmp(&left.get("updated_at_ms").and_then(Value::as_i64))
            .then_with(|| {
                left.get("id")
                    .and_then(Value::as_str)
                    .cmp(&right.get("id").and_then(Value::as_str))
            })
    });
    let truncated = sessions.len() > limit;
    sessions.truncate(limit);
    Ok(serde_json::json!({
        "ok": true,
        "collection": SESSIONS_COLLECTION,
        "count": sessions.len(),
        "truncated": truncated,
        "sessions": sessions,
    }))
}

fn handle_workjet_session_delete_command(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
    can_manage_all_records: bool,
) -> anyhow::Result<Value> {
    let owner_user_id = bounded_required(authorized_owner_user_id, "owner_user_id", 256)?;
    let session_id = session_id_for_delete(command)?;
    let conn = open_store(root)?;
    let existing = outbound_load_record(&conn, SESSIONS_COLLECTION, &session_id)?
        .context("Workjet session not found")?;
    ensure_owned(
        Some(&existing),
        &owner_user_id,
        "session",
        can_manage_all_records,
    )?;
    anyhow::ensure!(
        existing.get("active_transfer_id").is_none(),
        "cannot delete a session with an active transfer"
    );
    if existing.get("is_deleted").and_then(Value::as_bool) == Some(true) {
        return Ok(serde_json::json!({
            "ok": true,
            "collection": SESSIONS_COLLECTION,
            "session": existing,
        }));
    }
    let now = super::store::now_ms() as i64;
    let mut session = existing;
    session["is_deleted"] = Value::Bool(true);
    session["updated_at_ms"] = Value::from(now);
    let session = persist_and_project_idempotently(
        root,
        &conn,
        SESSIONS_COLLECTION,
        &session_id,
        now,
        session,
    )?;
    Ok(serde_json::json!({
        "ok": true,
        "collection": SESSIONS_COLLECTION,
        "session": session,
    }))
}

fn parse_create_payload(command: &BusinessCommand) -> anyhow::Result<SessionCreatePayload> {
    serde_json::from_value(command.payload.clone())
        .context("invalid ctox.workjet.session.create payload")
}

fn session_id_for_delete(command: &BusinessCommand) -> anyhow::Result<String> {
    let payload: SessionDeletePayload = serde_json::from_value(command.payload.clone())
        .context("invalid ctox.workjet.session.delete payload")?;
    let payload_id = payload
        .session_id
        .as_deref()
        .map(|value| bounded_required(value, "session_id", 160))
        .transpose()?;
    let record_id = command
        .record_id
        .as_deref()
        .map(|value| bounded_required(value, "record_id", 160))
        .transpose()?;
    if let (Some(payload_id), Some(record_id)) = (&payload_id, &record_id) {
        anyhow::ensure!(
            payload_id == record_id,
            "payload session_id and record_id must match"
        );
    }
    payload_id
        .or(record_id)
        .context("session_id or record_id is required")
}

fn ensure_asserted_session_id(
    command: &BusinessCommand,
    payload_session_id: Option<&str>,
    expected: &str,
) -> anyhow::Result<()> {
    for (field, asserted) in [
        ("session_id", payload_session_id),
        ("record_id", command.record_id.as_deref()),
    ] {
        if let Some(asserted) = asserted {
            let asserted = bounded_required(asserted, field, 160)?;
            anyhow::ensure!(
                asserted == expected,
                "{field} does not match deterministic Workjet session id"
            );
        }
    }
    Ok(())
}

fn deterministic_session_id(project_id: &str, working_copy_id: &str, owner: &str) -> String {
    let mut hasher = Sha256::new();
    for part in [project_id, working_copy_id, owner] {
        hasher.update((part.len() as u64).to_be_bytes());
        hasher.update(part.as_bytes());
    }
    format!("workjet_session_{:x}", hasher.finalize())
}

fn persist_idempotently(
    conn: &Connection,
    collection: &str,
    record_id: &str,
    now: i64,
    desired: Value,
) -> anyhow::Result<Value> {
    if let Some(existing) = outbound_load_record(conn, collection, record_id)? {
        if stable_record_content(&existing) == stable_record_content(&desired) {
            return Ok(existing);
        }
    }
    upsert_business_record(conn, collection, record_id, now, desired)?;
    outbound_load_record(conn, collection, record_id)?
        .with_context(|| format!("failed to reload {collection} record {record_id}"))
}

fn persist_and_project_idempotently(
    root: &Path,
    conn: &Connection,
    collection: &str,
    record_id: &str,
    now: i64,
    desired: Value,
) -> anyhow::Result<Value> {
    let record = persist_idempotently(conn, collection, record_id, now, desired)?;
    let updated_at_ms = record
        .get("updated_at_ms")
        .and_then(Value::as_i64)
        .unwrap_or(now);
    upsert_rxdb_collection_record(root, collection, record_id, updated_at_ms, record.clone())?;
    Ok(record)
}

fn stable_record_content(value: &Value) -> Value {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("_rev");
        object.remove("_deleted");
        object.remove("updated_at_ms");
    }
    value
}

fn ensure_owned(
    existing: Option<&Value>,
    owner_user_id: &str,
    kind: &str,
    can_manage_all_records: bool,
) -> anyhow::Result<()> {
    if let Some(existing) = existing {
        anyhow::ensure!(
            can_manage_all_records
                || existing.get("owner_user_id").and_then(Value::as_str) == Some(owner_user_id),
            "{kind} belongs to a different owner"
        );
    }
    Ok(())
}

fn bounded_required(value: &str, field: &str, max_chars: usize) -> anyhow::Result<String> {
    let value = value.trim();
    validate_bounded(value, field, max_chars, false)?;
    Ok(value.to_owned())
}

fn optional_bounded(
    value: Option<String>,
    field: &str,
    max_chars: usize,
) -> anyhow::Result<Option<String>> {
    value
        .map(|value| {
            let value = value.trim();
            validate_bounded(value, field, max_chars, true)?;
            Ok(value.to_owned())
        })
        .transpose()
}

fn validate_bounded(
    value: &str,
    field: &str,
    max_chars: usize,
    allow_empty: bool,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        allow_empty || !value.is_empty(),
        "{field} must not be empty"
    );
    anyhow::ensure!(
        value.chars().count() <= max_chars,
        "{field} exceeds {max_chars} characters"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "{field} contains control characters"
    );
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::business_os::store::{load_rxdb_collection_record, rxdb_store_path, CommandOrigin};
    use crate::business_os::store_workjet_projects::tests::create_workjet_rxdb_projection_tables;
    use rusqlite::Connection;
    use serde_json::json;
    use std::fs;
    use tempfile::tempdir;

    fn command(command_type: &str, record_id: Option<&str>, payload: Value) -> BusinessCommand {
        BusinessCommand {
            id: Some(format!("cmd-{command_type}")),
            module: "ctox".to_owned(),
            command_type: command_type.to_owned(),
            record_id: record_id.map(str::to_owned),
            payload,
            client_context: json!({}),
            origin: CommandOrigin::TrustedLocal,
        }
    }

    pub(crate) fn create_workjet_session_rxdb_projection_tables(root: &Path) -> anyhow::Result<()> {
        fs::create_dir_all(root.join("runtime"))?;
        let conn = Connection::open(rxdb_store_path(root))?;
        for collection in [SESSIONS_COLLECTION, TRANSFERS_COLLECTION] {
            conn.execute(
                &format!(
                    "CREATE TABLE IF NOT EXISTS ctox_business_os__{collection}__v0 (
                        id TEXT PRIMARY KEY NOT NULL,
                        revision TEXT,
                        deleted INTEGER NOT NULL DEFAULT 0,
                        lastWriteTime REAL NOT NULL DEFAULT 0,
                        data TEXT NOT NULL
                    )"
                ),
                [],
            )?;
        }
        Ok(())
    }

    fn seed_project_and_copy(root: &Path, owner: &str, suffix: &str) -> anyhow::Result<String> {
        super::super::store_workjet_projects::handle_workjet_project_upsert_command(
            root,
            &command(
                "ctox.workjet.project.upsert",
                None,
                json!({
                    "project_id": format!("project-{suffix}"),
                    "name": format!("Project {suffix}")
                }),
            ),
            owner,
        )?;
        let copy =
            super::super::store_workjet_projects::handle_workjet_working_copy_upsert_command(
                root,
                &command(
                    "ctox.workjet.working_copy.upsert",
                    None,
                    json!({
                        "project_id": format!("project-{suffix}"),
                        "computer_id": format!("computer-{suffix}"),
                        "path": format!("opaque://computer-{suffix}/project-{suffix}"),
                        "active": true
                    }),
                ),
                owner,
            )?;
        Ok(copy["working_copy"]["id"]
            .as_str()
            .context("working copy id")?
            .to_owned())
    }

    fn create_session(root: &Path, owner: &str, suffix: &str) -> anyhow::Result<Value> {
        let working_copy_id = seed_project_and_copy(root, owner, suffix)?;
        handle_workjet_session_create_command(
            root,
            &command(
                "ctox.workjet.session.create",
                None,
                json!({
                    "project_id": format!("project-{suffix}"),
                    "working_copy_id": working_copy_id,
                    "thread_id": format!("thread-{suffix}")
                }),
            ),
            owner,
        )
    }

    #[test]
    fn workjet_session_transfer_state_table_matches_rfc() {
        let states = WORKJET_SESSION_TRANSFER_STATES
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>();
        assert_eq!(states.len(), WORKJET_SESSION_TRANSFER_STATES.len());
        for transition in WORKJET_SESSION_TRANSFER_TRANSITIONS {
            assert!(states.contains(transition.from));
            assert!(states.contains(transition.to));
            assert_eq!(
                workjet_session_transfer_next_state(transition.from, transition.verb),
                Some(transition.to)
            );
        }

        for (state, verb) in [
            ("pause_requested", "pack_complete"),
            ("packing", "pause_ack"),
            ("packed", "apply_complete"),
            ("shipping", "confirm_working_copy"),
            ("applying", "resume_ack"),
            ("applied", "pack_complete"),
            ("switching", "pause_ack"),
            ("resuming", "confirm_working_copy"),
            ("aborting", "resume_ack"),
            ("completed", "abort"),
            ("rolled_back", "abort"),
            ("failed", "abort"),
        ] {
            assert_eq!(workjet_session_transfer_next_state(state, verb), None);
        }
    }

    #[test]
    fn workjet_session_create_is_idempotent_and_stamps_invariants() -> anyhow::Result<()> {
        let root = tempdir()?;
        create_workjet_rxdb_projection_tables(root.path())?;
        create_workjet_session_rxdb_projection_tables(root.path())?;
        let working_copy_id = seed_project_and_copy(root.path(), "owner-1", "one")?;
        let create = command(
            "ctox.workjet.session.create",
            None,
            json!({
                "project_id": "project-one",
                "working_copy_id": working_copy_id,
                "thread_id": "thread-1",
                "coding_session_id": "coding-1"
            }),
        );
        let first = handle_workjet_session_create_command(root.path(), &create, "owner-1")?;
        let second = handle_workjet_session_create_command(root.path(), &create, "owner-1")?;
        let session = &first["session"];
        assert!(session["id"]
            .as_str()
            .is_some_and(|id| id.starts_with("workjet_session_")));
        assert_eq!(session["working_copy_id"], working_copy_id);
        assert_eq!(session["computer_id"], "computer-one");
        assert_eq!(session["run_status"], "running");
        assert_eq!(session["fence_epoch"], 0);
        assert!(session.get("active_transfer_id").is_none());
        assert_eq!(session["_rev"], second["session"]["_rev"]);
        assert_eq!(session["updated_at_ms"], second["session"]["updated_at_ms"]);
        Ok(())
    }

    #[test]
    fn workjet_session_create_list_delete_are_owner_scoped_and_projected() -> anyhow::Result<()> {
        let root = tempdir()?;
        create_workjet_rxdb_projection_tables(root.path())?;
        create_workjet_session_rxdb_projection_tables(root.path())?;
        let owned = create_session(root.path(), "owner-1", "owned")?;
        create_session(root.path(), "owner-2", "foreign")?;
        let session_id = owned["session"]["id"]
            .as_str()
            .context("session id")?
            .to_owned();

        let listed = handle_workjet_session_list_command(
            root.path(),
            &command("ctox.workjet.session.list", None, json!({ "limit": 100 })),
            "owner-1",
        )?;
        assert_eq!(listed["count"], 1);
        assert_eq!(listed["sessions"][0]["id"], session_id);
        assert!(handle_workjet_session_delete_command(
            root.path(),
            &command("ctox.workjet.session.delete", Some(&session_id), json!({})),
            "owner-2",
            false,
        )
        .is_err());

        let delete = command("ctox.workjet.session.delete", Some(&session_id), json!({}));
        let first = handle_workjet_session_delete_command(root.path(), &delete, "owner-1", false)?;
        let second = handle_workjet_session_delete_command(root.path(), &delete, "owner-1", false)?;
        assert_eq!(first["session"]["is_deleted"], true);
        assert_eq!(first["session"]["_rev"], second["session"]["_rev"]);
        assert_eq!(
            load_rxdb_collection_record(root.path(), SESSIONS_COLLECTION, &session_id)?
                .context("session projection")?["is_deleted"],
            true
        );
        assert_eq!(
            handle_workjet_session_list_command(
                root.path(),
                &command("ctox.workjet.session.list", None, json!({})),
                "owner-1",
            )?["count"],
            0
        );
        Ok(())
    }

    #[test]
    fn workjet_session_rejects_spoofed_ids_unknown_fields_and_invalid_copy_binding(
    ) -> anyhow::Result<()> {
        let root = tempdir()?;
        let working_copy_id = seed_project_and_copy(root.path(), "owner-1", "strict")?;
        for payload in [
            json!({
                "project_id": "project-strict",
                "working_copy_id": working_copy_id,
                "owner_user_id": "attacker"
            }),
            json!({
                "project_id": "project-strict",
                "working_copy_id": working_copy_id,
                "session_id": "browser-chosen"
            }),
        ] {
            assert!(handle_workjet_session_create_command(
                root.path(),
                &command("ctox.workjet.session.create", None, payload),
                "owner-1",
            )
            .is_err());
        }
        assert!(handle_workjet_session_create_command(
            root.path(),
            &command(
                "ctox.workjet.session.create",
                None,
                json!({
                    "project_id": "project-strict",
                    "working_copy_id": working_copy_id
                })
            ),
            "owner-2",
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn workjet_session_owner_migration_updates_session_and_transfer_projections(
    ) -> anyhow::Result<()> {
        let root = tempdir()?;
        create_workjet_session_rxdb_projection_tables(root.path())?;
        let email = "owner@example.test";
        let stable = "stable-owner-id";
        let conn = open_store(root.path())?;
        let now = 10;
        let session = json!({
            "id": "workjet_session_email",
            "project_id": "project-email",
            "working_copy_id": "workjet_wc_email",
            "computer_id": "computer-email",
            "run_status": "running",
            "fence_epoch": 0,
            "owner_user_id": email,
            "created_at_ms": now,
            "updated_at_ms": now,
            "is_deleted": false
        });
        let transfer = json!({
            "id": "workjet_xfer_email",
            "session_id": "workjet_session_email",
            "project_id": "project-email",
            "source_working_copy_id": "workjet_wc_email",
            "source_computer_id": "computer-email",
            "target_computer_id": "computer-target",
            "target_path": "opaque://target/project-email",
            "state": "pause_requested",
            "fence_epoch": 1,
            "artifact_file_ids": [],
            "deadline_at_ms": 100,
            "created_at_ms": now,
            "updated_at_ms": now,
            "owner_user_id": email,
            "is_deleted": false
        });
        upsert_business_record(
            &conn,
            SESSIONS_COLLECTION,
            "workjet_session_email",
            now,
            session,
        )?;
        upsert_business_record(
            &conn,
            TRANSFERS_COLLECTION,
            "workjet_xfer_email",
            now,
            transfer,
        )?;
        drop(conn);

        migrate_signed_owner_alias(root.path(), stable, Some(email))?;
        for (collection, id) in [
            (SESSIONS_COLLECTION, "workjet_session_email"),
            (TRANSFERS_COLLECTION, "workjet_xfer_email"),
        ] {
            assert_eq!(
                outbound_load_record(&open_store(root.path())?, collection, id)?
                    .context("migrated core record")?["owner_user_id"],
                stable
            );
            assert_eq!(
                load_rxdb_collection_record(root.path(), collection, id)?
                    .context("migrated projected record")?["owner_user_id"],
                stable
            );
        }
        Ok(())
    }
}

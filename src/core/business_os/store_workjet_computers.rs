// Origin: CTOX
// License: AGPL-3.0-only

//! Native-owned Workjet computer assignment command handlers.
//!
//! A record in `workjet_computers` is an assignment to this CTOX instance.
//! `computer_id` is an opaque Workjet identity. CTOX must not derive it from,
//! or reinterpret it as, a hostname, environment, presentation, or path.

use super::store::{
    open_store, outbound_load_record, outbound_load_records_by_string_field,
    upsert_business_record, BusinessCommand,
};
use anyhow::Context;
use rusqlite::Connection;
use serde::Deserialize;
use serde_json::Value;
use std::collections::BTreeSet;
use std::path::Path;

pub(super) const COMPUTERS_COLLECTION: &str = "workjet_computers";
pub(super) const SELF_HOST_COLOCATION_CONFIRMATION: &str = "workjet-self-host-colocation.v1";

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComputerListPayload {
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComputerAssignPayload {
    computer_id: String,
    display_name: String,
    hosting_mode: String,
    #[serde(default)]
    capabilities: Vec<String>,
    #[serde(default)]
    self_hosted_colocation: bool,
    #[serde(default)]
    colocation_confirmation: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ComputerUnassignPayload {
    computer_id: String,
}

pub(super) fn handle_workjet_computer_store_command(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
) -> anyhow::Result<Value> {
    match command.command_type.as_str() {
        "ctox.workjet.computer.list" => {
            handle_workjet_computer_list_command(root, command, authorized_owner_user_id)
        }
        "ctox.workjet.computer.assign" => {
            handle_workjet_computer_assign_command(root, command, authorized_owner_user_id)
        }
        "ctox.workjet.computer.unassign" => {
            handle_workjet_computer_unassign_command(root, command, authorized_owner_user_id)
        }
        other => anyhow::bail!("unsupported Workjet computer command type: {other}"),
    }
}

fn handle_workjet_computer_list_command(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
) -> anyhow::Result<Value> {
    let payload: ComputerListPayload = serde_json::from_value(command.payload.clone())
        .context("invalid ctox.workjet.computer.list payload")?;
    let owner_user_id = bounded_required(authorized_owner_user_id, "owner_user_id", 256)?;
    let limit = payload.limit.unwrap_or(100).clamp(1, 100);
    let conn = open_store(root)?;
    let mut computers = outbound_load_records_by_string_field(
        &conn,
        COMPUTERS_COLLECTION,
        "owner_user_id",
        &owner_user_id,
    )?;
    computers.retain(|computer| {
        computer.get("status").and_then(Value::as_str) == Some("assigned")
            && computer.get("is_deleted").and_then(Value::as_bool) != Some(true)
    });
    computers.sort_by(|left, right| {
        left.get("display_name")
            .and_then(Value::as_str)
            .cmp(&right.get("display_name").and_then(Value::as_str))
            .then_with(|| {
                left.get("id")
                    .and_then(Value::as_str)
                    .cmp(&right.get("id").and_then(Value::as_str))
            })
    });
    let truncated = computers.len() > limit;
    Ok(serde_json::json!({
        "ok": true,
        "collection": COMPUTERS_COLLECTION,
        "count": computers.len().min(limit),
        "truncated": truncated,
    }))
}

fn handle_workjet_computer_assign_command(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
) -> anyhow::Result<Value> {
    let payload: ComputerAssignPayload = serde_json::from_value(command.payload.clone())
        .context("invalid ctox.workjet.computer.assign payload")?;
    let owner_user_id = bounded_required(authorized_owner_user_id, "owner_user_id", 256)?;
    let computer_id = bounded_required(&payload.computer_id, "computer_id", 160)?;
    let display_name = bounded_required(&payload.display_name, "display_name", 256)?;
    let hosting_mode = bounded_required(&payload.hosting_mode, "hosting_mode", 64)?;
    anyhow::ensure!(
        hosting_mode != "managed_backend",
        "managed backend hosts are backend-only and cannot be assigned as Workjet computers"
    );
    anyhow::ensure!(
        matches!(hosting_mode.as_str(), "workstation" | "self_hosted"),
        "unsupported Workjet computer hosting_mode"
    );
    if payload.self_hosted_colocation {
        anyhow::ensure!(
            hosting_mode == "self_hosted",
            "self-hosted co-location is valid only for self_hosted computers"
        );
        anyhow::ensure!(
            payload.colocation_confirmation.as_deref()
                == Some(SELF_HOST_COLOCATION_CONFIRMATION),
            "self-hosted co-location requires explicit workjet-self-host-colocation.v1 confirmation"
        );
    } else {
        anyhow::ensure!(
            payload.colocation_confirmation.is_none(),
            "co-location confirmation must be omitted while self-hosted co-location is disabled"
        );
    }
    anyhow::ensure!(
        payload.capabilities.len() <= 32,
        "capabilities exceeds 32 entries"
    );
    let capabilities = payload
        .capabilities
        .iter()
        .map(|value| bounded_required(value, "capability", 80))
        .collect::<anyhow::Result<BTreeSet<_>>>()?
        .into_iter()
        .collect::<Vec<_>>();

    let conn = open_store(root)?;
    let existing = outbound_load_record(&conn, COMPUTERS_COLLECTION, &computer_id)?;
    ensure_owned(existing.as_ref(), &owner_user_id)?;
    let now = super::store::now_ms() as i64;
    let created_at_ms = existing
        .as_ref()
        .and_then(|record| record.get("created_at_ms"))
        .and_then(Value::as_i64)
        .unwrap_or(now);
    let computer = serde_json::json!({
        "id": computer_id,
        "display_name": display_name,
        "hosting_mode": hosting_mode,
        "status": "assigned",
        "capabilities": capabilities,
        "self_hosted_colocation": payload.self_hosted_colocation,
        "owner_user_id": owner_user_id,
        "created_at_ms": created_at_ms,
        "updated_at_ms": now,
        "is_deleted": false,
    });
    let computer = persist_idempotently(&conn, COMPUTERS_COLLECTION, &computer_id, now, computer)?;
    Ok(serde_json::json!({
        "ok": true,
        "collection": COMPUTERS_COLLECTION,
        "computer": computer,
    }))
}

fn handle_workjet_computer_unassign_command(
    root: &Path,
    command: &BusinessCommand,
    authorized_owner_user_id: &str,
) -> anyhow::Result<Value> {
    let payload: ComputerUnassignPayload = serde_json::from_value(command.payload.clone())
        .context("invalid ctox.workjet.computer.unassign payload")?;
    let owner_user_id = bounded_required(authorized_owner_user_id, "owner_user_id", 256)?;
    let computer_id = bounded_required(&payload.computer_id, "computer_id", 160)?;
    let conn = open_store(root)?;
    let existing = outbound_load_record(&conn, COMPUTERS_COLLECTION, &computer_id)?
        .context("Workjet computer assignment not found")?;
    ensure_owned(Some(&existing), &owner_user_id)?;
    if existing.get("status").and_then(Value::as_str) == Some("unassigned") {
        return Ok(serde_json::json!({
            "ok": true,
            "collection": COMPUTERS_COLLECTION,
            "computer": existing,
        }));
    }
    let now = super::store::now_ms() as i64;
    let mut computer = existing;
    computer["status"] = Value::String("unassigned".to_owned());
    computer["updated_at_ms"] = Value::from(now);
    computer["unassigned_at_ms"] = Value::from(now);
    let computer = persist_idempotently(&conn, COMPUTERS_COLLECTION, &computer_id, now, computer)?;
    Ok(serde_json::json!({
        "ok": true,
        "collection": COMPUTERS_COLLECTION,
        "computer": computer,
    }))
}

pub(super) fn require_assigned_workjet_computer(
    conn: &Connection,
    computer_id: &str,
    owner_user_id: &str,
) -> anyhow::Result<Value> {
    let computer = outbound_load_record(conn, COMPUTERS_COLLECTION, computer_id)?
        .context("Workjet computer is not assigned to this CTOX instance")?;
    ensure_owned(Some(&computer), owner_user_id)?;
    anyhow::ensure!(
        computer.get("status").and_then(Value::as_str) == Some("assigned"),
        "Workjet computer is not assigned to this CTOX instance"
    );
    anyhow::ensure!(
        computer.get("hosting_mode").and_then(Value::as_str) != Some("managed_backend"),
        "managed backend hosts are backend-only and cannot run Workjet workers"
    );
    Ok(computer)
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

fn stable_record_content(value: &Value) -> Value {
    let mut value = value.clone();
    if let Some(object) = value.as_object_mut() {
        object.remove("_rev");
        object.remove("_deleted");
        object.remove("updated_at_ms");
    }
    value
}

fn ensure_owned(existing: Option<&Value>, owner_user_id: &str) -> anyhow::Result<()> {
    if let Some(existing) = existing {
        anyhow::ensure!(
            existing.get("owner_user_id").and_then(Value::as_str) == Some(owner_user_id),
            "Workjet computer belongs to a different owner"
        );
    }
    Ok(())
}

fn bounded_required(value: &str, field: &str, max_chars: usize) -> anyhow::Result<String> {
    let value = value.trim();
    anyhow::ensure!(!value.is_empty(), "{field} must not be empty");
    anyhow::ensure!(
        value.chars().count() <= max_chars,
        "{field} exceeds {max_chars} characters"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "{field} contains control characters"
    );
    Ok(value.to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::business_os::store::CommandOrigin;
    use serde_json::json;
    use tempfile::tempdir;

    fn command(command_type: &str, payload: Value) -> BusinessCommand {
        BusinessCommand {
            id: Some("cmd-workjet-computer-test".to_owned()),
            module: "ctox".to_owned(),
            command_type: command_type.to_owned(),
            record_id: None,
            payload,
            client_context: json!({}),
            origin: CommandOrigin::TrustedLocal,
        }
    }

    fn assign(root: &Path, computer_id: &str) -> anyhow::Result<Value> {
        handle_workjet_computer_assign_command(
            root,
            &command(
                "ctox.workjet.computer.assign",
                json!({
                    "computer_id": computer_id,
                    "display_name": "Workstation",
                    "hosting_mode": "workstation",
                    "capabilities": ["codex", "claude", "codex"]
                }),
            ),
            "owner-1",
        )
    }

    #[test]
    fn assign_is_owner_scoped_idempotent_and_opaque() -> anyhow::Result<()> {
        let root = tempdir()?;
        let computer_id = "workjet_device_8J7tQvKx-not-a-hostname";
        let first = assign(root.path(), computer_id)?;
        let second = assign(root.path(), computer_id)?;
        assert_eq!(first["computer"]["id"], computer_id);
        assert_eq!(first["computer"]["_rev"], second["computer"]["_rev"]);
        assert_eq!(
            first["computer"]["capabilities"],
            json!(["claude", "codex"])
        );
        assert_eq!(first["computer"]["self_hosted_colocation"], false);
        assert!(first["computer"].get("hostname").is_none());
        assert!(first["computer"].get("environment_id").is_none());
        assert!(first["computer"].get("presentation_id").is_none());
        assert!(handle_workjet_computer_assign_command(
            root.path(),
            &command(
                "ctox.workjet.computer.assign",
                json!({
                    "computer_id": computer_id,
                    "display_name": "Other owner",
                    "hosting_mode": "workstation"
                }),
            ),
            "owner-2",
        )
        .is_err());
        Ok(())
    }

    #[test]
    fn managed_backend_is_always_rejected() -> anyhow::Result<()> {
        let root = tempdir()?;
        let result = handle_workjet_computer_assign_command(
            root.path(),
            &command(
                "ctox.workjet.computer.assign",
                json!({
                    "computer_id": "backend-1",
                    "display_name": "Managed backend",
                    "hosting_mode": "managed_backend"
                }),
            ),
            "owner-1",
        );
        assert!(result.is_err());
        assert!(outbound_load_record(
            &open_store(root.path())?,
            COMPUTERS_COLLECTION,
            "backend-1"
        )?
        .is_none());
        Ok(())
    }

    #[test]
    fn assignment_rejects_identity_heuristics_and_unknown_fields() -> anyhow::Result<()> {
        let root = tempdir()?;
        for forbidden in [
            json!({ "hostname": "mac.local" }),
            json!({ "environment_id": "primary" }),
            json!({ "presentation_id": "welsch" }),
        ] {
            let mut payload = json!({
                "computer_id": "opaque-1",
                "display_name": "Current Mac",
                "hosting_mode": "workstation"
            });
            payload
                .as_object_mut()
                .expect("payload object")
                .extend(forbidden.as_object().expect("forbidden object").clone());
            assert!(handle_workjet_computer_assign_command(
                root.path(),
                &command("ctox.workjet.computer.assign", payload),
                "owner-1",
            )
            .is_err());
        }
        Ok(())
    }

    #[test]
    fn self_hosted_colocation_requires_exact_explicit_confirmation() -> anyhow::Result<()> {
        let root = tempdir()?;
        for confirmation in [None, Some("yes"), Some("workjet-self-host-colocation.v0")] {
            let result = handle_workjet_computer_assign_command(
                root.path(),
                &command(
                    "ctox.workjet.computer.assign",
                    json!({
                        "computer_id": "self-hosted-1",
                        "display_name": "Self-hosted",
                        "hosting_mode": "self_hosted",
                        "self_hosted_colocation": true,
                        "colocation_confirmation": confirmation
                    }),
                ),
                "owner-1",
            );
            assert!(result.is_err());
        }
        let accepted = handle_workjet_computer_assign_command(
            root.path(),
            &command(
                "ctox.workjet.computer.assign",
                json!({
                    "computer_id": "self-hosted-1",
                    "display_name": "Self-hosted",
                    "hosting_mode": "self_hosted",
                    "self_hosted_colocation": true,
                    "colocation_confirmation": SELF_HOST_COLOCATION_CONFIRMATION
                }),
            ),
            "owner-1",
        )?;
        assert_eq!(accepted["computer"]["self_hosted_colocation"], true);
        Ok(())
    }

    #[test]
    fn list_and_unassign_are_bounded_owner_scoped_and_idempotent() -> anyhow::Result<()> {
        let root = tempdir()?;
        assign(root.path(), "computer-1")?;
        handle_workjet_computer_assign_command(
            root.path(),
            &command(
                "ctox.workjet.computer.assign",
                json!({
                    "computer_id": "computer-2",
                    "display_name": "Other",
                    "hosting_mode": "workstation"
                }),
            ),
            "owner-2",
        )?;
        let listed = handle_workjet_computer_list_command(
            root.path(),
            &command("ctox.workjet.computer.list", json!({ "limit": 100 })),
            "owner-1",
        )?;
        assert_eq!(listed["count"], 1);
        assert!(listed.get("computers").is_none());

        let unassign = command(
            "ctox.workjet.computer.unassign",
            json!({ "computer_id": "computer-1" }),
        );
        let first = handle_workjet_computer_unassign_command(root.path(), &unassign, "owner-1")?;
        let second = handle_workjet_computer_unassign_command(root.path(), &unassign, "owner-1")?;
        assert_eq!(first["computer"]["status"], "unassigned");
        assert_eq!(first["computer"]["_rev"], second["computer"]["_rev"]);
        let listed = handle_workjet_computer_list_command(
            root.path(),
            &command("ctox.workjet.computer.list", json!({})),
            "owner-1",
        )?;
        assert_eq!(listed["count"], 0);
        Ok(())
    }
}

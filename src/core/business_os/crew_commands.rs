use super::session::{session_user_id, BusinessOsSession};
use super::store::BusinessCommand;
use crate::crew::{self, Soul, Specialties};
use anyhow::{ensure, Context, Result};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::path::Path;

fn text<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    let value = value
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .with_context(|| format!("{key} is required"))?;
    ensure!(value.chars().count() <= 200, "{key} too long");
    Ok(value)
}
fn specialties(value: Value) -> Result<Specialties> {
    let value: Specialties = serde_json::from_value(value)?;
    for list in [
        &value.modules,
        &value.command_types,
        &value.skills,
        &value.tags,
    ] {
        ensure!(
            list.len() <= 20 && list.iter().all(|s| crew::safe_prose(s, 100)),
            "invalid specialties"
        );
    }
    Ok(value)
}
fn audit_payload(payload: &Value) -> Value {
    let mut out = json!({});
    for key in [
        "member_id",
        "learning_id",
        "task_id",
        "name",
        "shape",
        "color",
    ] {
        if let Some(value) = payload.get(key).and_then(Value::as_str) {
            if crew::safe_prose(value, 200) {
                out[key] = json!(value);
            }
        }
    }
    if let Some(value) = payload.get("archived").and_then(Value::as_bool) {
        out["archived"] = json!(value)
    }
    out
}
pub(super) fn control(
    root: &Path,
    command: &BusinessCommand,
    session: &BusinessOsSession,
) -> Result<Value> {
    crate::service::harness_flow::record_harness_flow_event(
        root,
        crate::service::harness_flow::RecordHarnessFlowEventRequest {
            event_kind: "crew.control",
            title: &command.command_type,
            body_text: "",
            message_key: None,
            work_id: None,
            ticket_key: None,
            attempt_index: None,
            metadata: json!({"actor":session_user_id(session),"command_type":command.command_type,"stage":"requested","payload":audit_payload(&command.payload)}),
        },
    )?;
    // Central authorization and command receipts precede this core transaction.
    let outcome = (|| -> Result<Value> {
        let conn = Connection::open(crate::paths::core_db(root))?;
        conn.busy_timeout(crate::persistence::sqlite_busy_timeout_duration())?;
        let tx = conn.unchecked_transaction()?;
        let p = &command.payload;
        let now = chrono::Utc::now().to_rfc3339();
        let result = match command.command_type.as_str() {
            "ctox.crew.member.create" => {
                let name = text(p, "name")?;
                ensure!(crew::safe_prose(name, 60), "invalid member name");
                let shape = text(p, "shape")?;
                ensure!(
                    ["round", "square", "triangle", "blob"].contains(&shape),
                    "invalid shape"
                );
                let color = text(p, "color")?;
                ensure!(
                    ["#1685ee", "#00aa9a", "#7d7f84", "#7c6df2", "#e97255", "#34a26f"]
                        .contains(&color),
                    "color is not a CREW_COLOR"
                );
                let soul: Soul =
                    serde_json::from_value(p.get("soul").cloned().context("soul is required")?)?;
                soul.validate()?;
                let specialties = specialties(p.get("specialties").cloned().unwrap_or(json!({})))?;
                let count: i64 =
                    tx.query_row("SELECT COUNT(*) FROM crew_members", [], |r| r.get(0))?;
                ensure!(count < 1000, "crew member capacity reached");
                let id = format!("crew:{}", uuid::Uuid::new_v4());
                tx.execute("INSERT INTO crew_members(id,name,shape,color,created_at,archived,soul_json,specialties_json,stats_json,updated_at)
                VALUES(?1,?2,?3,?4,?5,0,?6,?7,?8,?5)",params![id,name,shape,color,now,serde_json::to_string(&soul)?,serde_json::to_string(&specialties)?,serde_json::to_string(&crew::Stats::default())?])?;
                json!({"member_id":id})
            }
            "ctox.crew.member.update" => {
                let id = text(p, "member_id")?;
                let mut member = crew::members(&tx)?
                    .into_iter()
                    .find(|m| m.id == id)
                    .context("member not found")?;
                if let Some(name) = p.get("name") {
                    let name = name.as_str().context("invalid name")?;
                    ensure!(crew::safe_prose(name, 60), "invalid name");
                    member.name = name.trim().into();
                }
                if let Some(soul) = p.get("soul") {
                    member.soul = serde_json::from_value(soul.clone())?;
                    member.soul.validate()?;
                }
                if let Some(value) = p.get("specialties") {
                    member.specialties = specialties(value.clone())?;
                }
                if let Some(archived) = p.get("archived") {
                    member.archived = archived.as_bool().context("archived must be boolean")?;
                }
                tx.execute("UPDATE crew_members SET name=?2,soul_json=?3,specialties_json=?4,archived=?5,updated_at=?6 WHERE id=?1",params![id,member.name,serde_json::to_string(&member.soul)?,serde_json::to_string(&member.specialties)?,member.archived,now])?;
                json!({"member_id":id})
            }
            "ctox.crew.assign" => {
                let task = text(p, "task_id")?;
                let member = text(p, "member_id")?;
                let exists: bool = tx.query_row(
                    "SELECT EXISTS(SELECT 1 FROM crew_members WHERE id=?1 AND archived=0)",
                    [member],
                    |r| r.get(0),
                )?;
                ensure!(exists, "active member not found");
                let changed=tx.execute("UPDATE communication_routing_state SET crew_member_id=?2,updated_at=?3
                WHERE message_key=?1 AND route_status IN ('pending','blocked') AND lease_owner IS NULL",params![task,member,now])?;
                ensure!(
                    changed == 1,
                    "assignment requires an unleased pending or blocked task"
                );
                json!({"task_id":task,"member_id":member})
            }
            "ctox.crew.learning.confirm"
            | "ctox.crew.learning.update"
            | "ctox.crew.learning.delete" => {
                let id = text(p, "learning_id")?;
                let member: Option<String> = tx
                    .query_row(
                        "SELECT member_id FROM crew_member_learnings WHERE id=?1",
                        [id],
                        |r| r.get(0),
                    )
                    .optional()?;
                ensure!(member.is_some(), "learning not found");
                match command.command_type.as_str() {
                    "ctox.crew.learning.confirm" => {
                        tx.execute(
                            "UPDATE crew_member_learnings SET confirmed_by_owner=1 WHERE id=?1",
                            [id],
                        )?;
                    }
                    "ctox.crew.learning.delete" => {
                        tx.execute("DELETE FROM crew_member_learnings WHERE id=?1", [id])?;
                    }
                    _ => {
                        if let Some(value) = p.get("text") {
                            let value = value.as_str().context("text must be a string")?;
                            ensure!(crew::safe_prose(value, 400), "invalid learning text");
                            tx.execute("UPDATE crew_member_learnings SET text=?2,normalized_text=?3 WHERE id=?1",params![id,value.trim(),crew::normalized(value)])?;
                        }
                        if let Some(archived) = p.get("archived") {
                            tx.execute(
                                "UPDATE crew_member_learnings SET archived=?2 WHERE id=?1",
                                params![
                                    id,
                                    archived.as_bool().context("archived must be boolean")?
                                ],
                            )?;
                        }
                    }
                }
                json!({"learning_id":id})
            }
            _ => anyhow::bail!("unsupported crew command"),
        };
        tx.commit()?;
        super::harness_cockpit::refresh_after_finalization(&crate::paths::core_db(root));
        Ok(json!({"ok":true,"result":result}))
    })();
    crate::service::harness_flow::record_harness_flow_event_lossy(
        root,
        crate::service::harness_flow::RecordHarnessFlowEventRequest {
            event_kind: "crew.control",
            title: &command.command_type,
            body_text: "",
            message_key: None,
            work_id: None,
            ticket_key: None,
            attempt_index: None,
            metadata: json!({"actor":session_user_id(session),"command_type":command.command_type,
            "stage":if outcome.is_ok(){"succeeded"}else{"failed"},"ok":outcome.is_ok(),
            "payload":audit_payload(&command.payload)}),
        },
    );
    outcome
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn bringup_never_creates_crew_grants() -> Result<()> {
        let root = tempfile::tempdir()?;
        super::super::store::ensure_legacy_collection_grants(
            root.path(),
            &["ctox_crew_members".into(), "ctox_crew_learnings".into()],
        )?;
        let conn = super::super::store::open_store(root.path())?;
        let count:i64=conn.query_row("SELECT COUNT(*) FROM business_permission_grants WHERE scope_id IN ('ctox_crew_members','ctox_crew_learnings')",[],|r|r.get(0))?;
        assert_eq!(count, 0);
        Ok(())
    }
    #[test]
    fn requested_audit_never_copies_arbitrary_payload() {
        let audit = audit_payload(
            &json!({"member_id":"crew-milo","soul":{"secret":"hidden"},"foreign":"x".repeat(1_048_576)}),
        );
        assert_eq!(audit, json!({"member_id":"crew-milo"}));
    }
}

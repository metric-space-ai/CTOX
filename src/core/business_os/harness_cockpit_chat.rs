//! Chat lifecycle messages are server projections, never harness-authored chat rows.
#[cfg(test)]
#[path = "harness_cockpit_chat_tests.rs"]
mod tests;
use crate::business_os::{store, store_projections};
use crate::mission::channels;
use anyhow::Result;
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::path::Path;

pub(crate) fn trim_messages(messages: &mut Vec<Value>) {
    while messages.len() > 40 {
        let position = messages
            .iter()
            .position(|message| message.get("kind").and_then(Value::as_str) == Some("status"))
            .unwrap_or(0);
        messages.remove(position);
    }
}

// origin/main has no resolve_chat_reply_language helper yet. Explicit persisted
// language/locale wins; the established Business OS chat fallback is German.
fn resolve_chat_reply_language(command: &store::BusinessCommand) -> &'static str {
    let language = [&command.payload, &command.client_context]
        .into_iter()
        .find_map(|value| {
            ["reply_language", "language", "locale"]
                .into_iter()
                .find_map(|key| value.get(key).and_then(Value::as_str))
        })
        .unwrap_or("de");
    if language.to_ascii_lowercase().starts_with("en") {
        "en"
    } else {
        "de"
    }
}

fn text(language: &str, key: &str) -> &'static str {
    match (language, key) {
        ("en", "queued") => {
            "Task added to the CTOX queue. Progress and the reply will appear here."
        }
        (_, "queued") => {
            "Aufgabe in der CTOX Queue angelegt. Fortschritt und Antwort erscheinen hier."
        }
        ("en", "leased") => "Work started.",
        (_, "leased") => "Die Bearbeitung hat begonnen.",
        ("en", "plan") => "Plan updated",
        (_, "plan") => "Plan aktualisiert",
        ("en", "retry_wait") => "The attempt will be retried",
        (_, "retry_wait") => "Der Versuch wird wiederholt",
        ("en", "blocked") => "Work is paused until the cause is resolved",
        (_, "blocked") => "Die Bearbeitung wartet, bis die Ursache behoben ist",
        ("en", "review_rework") => "Review requested changes; the task will be revised",
        (_, "review_rework") => "Die Prüfung verlangt Änderungen; die Aufgabe wird überarbeitet",
        _ => "",
    }
}

pub(crate) fn queued_chat_text(command: &store::BusinessCommand) -> &'static str {
    text(resolve_chat_reply_language(command), "queued")
}

pub(super) fn project(root: &Path, core: &Connection) -> Result<()> {
    let business = store::open_store(root)?;
    let mut writer = store::BusinessProjectionWriter::open(root)?;
    business.execute_batch("CREATE TABLE IF NOT EXISTS cockpit_chat_delivery(command_id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cockpit_chat_message_delivery(command_id TEXT NOT NULL, message_id TEXT NOT NULL, PRIMARY KEY(command_id,message_id));")?;
    // `terminal` marks a delivery made after the task reached a terminal route
    // status: nothing about that chat changes any more, so later passes skip it
    // before the expensive projection/progress lookups. Without this every pass
    // re-projected all 300 retained terminal chats (thesen 07.09.2026:
    // project_chat 190–345 s per pass, browser replication starved).
    let _ = business.execute(
        "ALTER TABLE cockpit_chat_delivery ADD COLUMN terminal INTEGER NOT NULL DEFAULT 0",
        [],
    );
    // Only materialized chats are candidates; projection never creates chats for background work.
    let mut cursor = String::new();
    loop {
        let mut statement=business.prepare("SELECT command_id FROM business_commands WHERE command_id>?1 AND command_type='business_os.chat.task' AND (status NOT IN ('completed','failed','cancelled') OR command_id IN (SELECT command_id FROM business_commands WHERE command_type='business_os.chat.task' AND status IN ('completed','failed','cancelled') ORDER BY observed_at_ms DESC,command_id DESC LIMIT 300)) ORDER BY command_id LIMIT 128")?;
        let commands = statement
            .query_map([&cursor], |row| row.get::<_, String>(0))?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        if commands.is_empty() {
            break;
        }
        for command_id in commands {
            cursor = command_id.clone();
            let settled: bool = business
                .query_row(
                    "SELECT terminal FROM cockpit_chat_delivery WHERE command_id=?1",
                    [&command_id],
                    |row| row.get::<_, i64>(0),
                )
                .optional()?
                .is_some_and(|flag| flag != 0);
            if settled {
                continue;
            }
            let command = store::load_business_command(&business, &command_id)?;
            let chat_id = store_projections::business_chat_id(&command, &command_id);
            let raw:Option<String>=business.query_row("SELECT payload_json FROM business_records WHERE collection='business_chats' AND record_id=?1 AND deleted=0",[&chat_id],|row|row.get(0)).optional()?;
            let Some(raw) = raw else {
                continue;
            };
            let mut chat: Value = serde_json::from_str(&raw)?;
            let projection = channels::business_command_projection(root, &command_id)?;
            let Some(task_id) = projection
                .get("task_id")
                .and_then(Value::as_str)
                .filter(|id| !id.is_empty())
            else {
                continue;
            };
            let Some(task) = channels::load_queue_task(root, task_id)? else {
                continue;
            };
            let run_id:Option<String>=core.query_row("SELECT json_extract(metadata_json,'$.attempt_id') FROM ctox_harness_flow_events WHERE message_key=?1 AND json_extract(metadata_json,'$.attempt_id') IS NOT NULL ORDER BY created_at DESC LIMIT 1",[task_id],|row|row.get(0)).optional()?;
            let progress = crate::lcm::run_task_execution_progress_for_task(
                &crate::paths::core_db(root),
                task_id,
            )?;
            let language = resolve_chat_reply_language(&command);
            let mut additions = Vec::new();
            let terminal = matches!(
                task.route_status.as_str(),
                "handled" | "failed" | "cancelled"
            );
            let status = if !terminal
                && projection.get("execution_phase").and_then(Value::as_str) == Some("retry_wait")
            {
                "retry_wait"
            } else {
                task.route_status.as_str()
            };
            let note = task
                .hold_reason
                .as_deref()
                .or(task.status_note.as_deref())
                .unwrap_or("");
            // The durable lease counter also covers a slice that finished before delivery.
            if task.attempt > 0 {
                additions.push(("status", text(language, "leased").to_string()));
            }
            if matches!(status, "retry_wait" | "blocked" | "review_rework") {
                let mut message = text(language, status).to_string();
                if status != "leased" && !note.is_empty() {
                    message.push_str(": ");
                    message.push_str(note);
                }
                if status == "retry_wait" {
                    if let Some(next) = &task.retry_not_before {
                        message.push_str(" — ");
                        message.push_str(next);
                    }
                }
                additions.push(("status", message));
            }
            if progress.is_some() {
                // Replay every retained revision, not just the latest coalesced snapshot.
                // Forty is the chat's own message cap; older status messages cannot survive it.
                let mut plans = core.prepare("SELECT revision,steps_json FROM (SELECT revision,steps_json FROM task_execution_plan_revisions WHERE task_id=?1 AND (?2 IS NULL OR attempt_id=?2) ORDER BY revision DESC LIMIT 40) ORDER BY revision")?;
                let plans = plans
                    .query_map(params![task_id, run_id], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                    })?
                    .collect::<rusqlite::Result<Vec<_>>>()?;
                for (revision, steps) in plans {
                    let mut message = format!("{} ({revision})", text(language, "plan"));
                    let steps: Vec<Value> = serde_json::from_str(&steps)?;
                    if let Some((index, step)) = steps.iter().enumerate().find(|(_, step)| {
                        step.get("status").and_then(Value::as_str) == Some("in_progress")
                    }) {
                        let position = index + 1;
                        let step_word = if language == "en" { "Step" } else { "Schritt" };
                        message.push_str(&format!(": {step_word} {position}"));
                        if let Some(title) = step.get("label").and_then(Value::as_str) {
                            message.push_str(" — ");
                            message.push_str(title);
                        }
                    }
                    additions.push(("status", message));
                }
            }
            if !terminal {
                if let Some(message) = projection
                    .pointer("/result/user_message")
                    .and_then(Value::as_str)
                    .filter(|s| !s.trim().is_empty())
                {
                    let kind = if projection
                        .pointer("/result/requires_input")
                        .and_then(Value::as_bool)
                        == Some(true)
                    {
                        "question"
                    } else {
                        "interim"
                    };
                    additions.push((kind, message.to_string()));
                }
            }
            // Receipt survives status-first trimming, so maintenance cannot resurrect old messages.
            // Record it only after delivery to both stores; partial delivery remains retryable.
            let fingerprint = channels::stable_digest(&serde_json::to_string(&json!({
                "task_id":task_id,"attempt":task.attempt,"run_id":run_id,"messages":additions
            }))?);
            let delivered: Option<String> = business
                .query_row(
                    "SELECT fingerprint FROM cockpit_chat_delivery WHERE command_id=?1",
                    [&command_id],
                    |row| row.get(0),
                )
                .optional()?;
            if delivered.as_deref() == Some(&fingerprint) {
                if terminal {
                    business.execute(
                        "UPDATE cockpit_chat_delivery SET terminal=1 WHERE command_id=?1",
                        [&command_id],
                    )?;
                }
                continue;
            }
            let Some(messages) = chat.get_mut("messages").and_then(Value::as_array_mut) else {
                continue;
            };
            if terminal {
                for message in messages
                    .iter_mut()
                    .filter(|m| m["kind"] == "reply" && m["command_id"] == command_id)
                {
                    if message.get("run_id").is_none_or(Value::is_null) {
                        message["run_id"] = json!(run_id);
                    }
                }
            }
            let now = chrono::Utc::now().timestamp_millis();
            let mut message_receipts = Vec::new();
            for (kind, message) in additions {
                let digest = channels::stable_digest(&format!(
                    "{task_id}:{}:{kind}:{message}",
                    task.attempt
                ));
                let id = format!("cockpit_{digest}");
                message_receipts.push(id.clone());
                if let Some(existing) = messages
                    .iter_mut()
                    .find(|m| m.get("id").and_then(Value::as_str) == Some(&id))
                {
                    existing["run_id"] = json!(run_id);
                    continue;
                }
                let already_delivered: bool = business.query_row("SELECT EXISTS(SELECT 1 FROM cockpit_chat_message_delivery WHERE command_id=?1 AND message_id=?2)",params![command_id,id], |row|row.get(0))?;
                if already_delivered {
                    continue;
                }
                messages.push(json!({"id":id,"role":"ctox","kind":kind,"text":message,
                    "task_id":task_id,"command_id":command_id,"run_id":run_id,
                    "taskId":task_id,"commandId":command_id,"status":status,"createdAt":now}));
            }
            {
                trim_messages(messages);
                chat["updated_at_ms"] = json!(now);
                writer.upsert_source_projection("business_chats", &chat_id, now, chat)?;
                if writer.delivered_to_rxdb("business_chats") {
                    let tx = business.unchecked_transaction()?;
                    for id in message_receipts {
                        tx.execute("INSERT OR IGNORE INTO cockpit_chat_message_delivery(command_id,message_id) VALUES(?1,?2)",params![command_id,id])?;
                    }
                    tx.execute("INSERT INTO cockpit_chat_delivery(command_id,fingerprint,terminal) VALUES(?1,?2,?3) ON CONFLICT(command_id) DO UPDATE SET fingerprint=excluded.fingerprint, terminal=excluded.terminal",params![command_id,fingerprint,i64::from(terminal)])?;
                    tx.commit()?;
                }
            }
        }
    }
    business.execute("DELETE FROM cockpit_chat_delivery WHERE command_id NOT IN (SELECT command_id FROM business_commands WHERE command_type='business_os.chat.task' AND (status NOT IN ('completed','failed','cancelled') OR command_id IN (SELECT command_id FROM business_commands WHERE command_type='business_os.chat.task' AND status IN ('completed','failed','cancelled') ORDER BY observed_at_ms DESC,command_id DESC LIMIT 300)))",[])?;
    business.execute("DELETE FROM cockpit_chat_message_delivery WHERE command_id NOT IN (SELECT command_id FROM cockpit_chat_delivery)",[])?;
    Ok(())
}

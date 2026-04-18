use anyhow::Context;
use anyhow::Result;
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use serde::Serialize;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;

use crate::channels;
use crate::tickets;

const DEFAULT_LIST_LIMIT: usize = 20;
const DEFAULT_TICKET_SYSTEM: &str = "internal";
const SPILL_RESTORE_LEASE_OWNER: &str = "spill-restore-hold";
const SPILL_RESTORE_TITLE_PREFIX: &str = "spill restore: ";

#[derive(Debug, Clone, Serialize)]
struct QueueTicketBridgeView {
    message_key: String,
    work_id: String,
    ticket_system: String,
    bridge_state: String,
    spilled_at: String,
    restored_at: Option<String>,
    task: channels::QueueTaskView,
    ticket: tickets::TicketSelfWorkItemView,
}

#[derive(Debug, Clone, Serialize)]
struct QueueTicketBridgeListItem {
    message_key: String,
    work_id: String,
    ticket_system: String,
    bridge_state: String,
    spilled_at: String,
    restored_at: Option<String>,
    task: Option<channels::QueueTaskView>,
    ticket: Option<tickets::TicketSelfWorkItemView>,
}

#[derive(Debug, Clone, Serialize)]
struct QueueSpillCandidateView {
    message_key: String,
    priority: String,
    route_status: String,
    title: String,
    thread_key: String,
    suggested_skill: Option<String>,
    workspace_root: Option<String>,
    candidate_score: i64,
    recommendation: String,
    reasons: Vec<String>,
}

pub fn handle_queue_command(root: &Path, args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("");
    match command {
        "add" => {
            let title = required_flag_value(args, "--title")
                .context("usage: ctox queue add --title <label> --prompt <text> [--thread-key <key>] [--workspace-root <path>] [--skill <name>] [--priority <urgent|high|normal|low>] [--parent-message-key <key>]")?;
            let prompt = required_flag_value(args, "--prompt")
                .context("usage: ctox queue add --title <label> --prompt <text> [--thread-key <key>] [--workspace-root <path>] [--skill <name>] [--priority <urgent|high|normal|low>] [--parent-message-key <key>]")?;
            let thread_key = find_flag_value(args, "--thread-key")
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| default_thread_key(title));
            let task = channels::create_queue_task(
                root,
                channels::QueueTaskCreateRequest {
                    title: title.to_string(),
                    prompt: prompt.to_string(),
                    thread_key,
                    workspace_root: find_flag_value(args, "--workspace-root")
                        .map(ToOwned::to_owned)
                        .or_else(|| channels::legacy_workspace_root_from_prompt(prompt)),
                    priority: find_flag_value(args, "--priority")
                        .unwrap_or("normal")
                        .to_string(),
                    suggested_skill: find_flag_value(args, "--skill").map(ToOwned::to_owned),
                    parent_message_key: find_flag_value(args, "--parent-message-key")
                        .map(ToOwned::to_owned),
                },
            )?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "list" => {
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_LIST_LIMIT);
            let statuses = collect_flag_values(args, "--status");
            let tasks = channels::list_queue_tasks(root, &statuses, limit)?;
            print_json(&json!({
                "ok": true,
                "count": tasks.len(),
                "tasks": tasks,
            }))
        }
        "show" => {
            let message_key = required_flag_value(args, "--message-key")
                .or_else(|| args.get(1).map(String::as_str))
                .context("usage: ctox queue show --message-key <key>")?;
            let task = channels::load_queue_task(root, message_key)?.context("queue task not found")?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "edit" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue edit --message-key <key> [--title <label>] [--prompt <text>] [--thread-key <key>] [--workspace-root <path>] [--clear-workspace-root] [--skill <name>] [--clear-skill] [--priority <urgent|high|normal|low>]")?;
            ensure_edit_requested(
                args,
                &["--title", "--prompt", "--thread-key", "--workspace-root", "--skill", "--priority"],
                &["--clear-skill", "--clear-workspace-root"],
            )?;
            let task = channels::update_queue_task(
                root,
                channels::QueueTaskUpdateRequest {
                    message_key: message_key.to_string(),
                    title: find_flag_value(args, "--title").map(ToOwned::to_owned),
                    prompt: find_flag_value(args, "--prompt").map(ToOwned::to_owned),
                    thread_key: find_flag_value(args, "--thread-key").map(ToOwned::to_owned),
                    workspace_root: find_flag_value(args, "--workspace-root").map(ToOwned::to_owned),
                    clear_workspace_root: args.iter().any(|arg| arg == "--clear-workspace-root"),
                    priority: find_flag_value(args, "--priority").map(ToOwned::to_owned),
                    suggested_skill: find_flag_value(args, "--skill").map(ToOwned::to_owned),
                    clear_skill: args.iter().any(|arg| arg == "--clear-skill"),
                    route_status: None,
                    status_note: None,
                    clear_note: false,
                },
            )?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "reprioritize" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue reprioritize --message-key <key> --priority <urgent|high|normal|low>")?;
            let priority = required_flag_value(args, "--priority")
                .context("usage: ctox queue reprioritize --message-key <key> --priority <urgent|high|normal|low>")?;
            let task = channels::update_queue_task(
                root,
                channels::QueueTaskUpdateRequest {
                    message_key: message_key.to_string(),
                    priority: Some(priority.to_string()),
                    ..Default::default()
                },
            )?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "block" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue block --message-key <key> --reason <text>")?;
            let reason = required_flag_value(args, "--reason")
                .context("usage: ctox queue block --message-key <key> --reason <text>")?;
            let task = channels::update_queue_task(
                root,
                channels::QueueTaskUpdateRequest {
                    message_key: message_key.to_string(),
                    route_status: Some("blocked".to_string()),
                    status_note: Some(reason.to_string()),
                    ..Default::default()
                },
            )?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "release" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue release --message-key <key> [--priority <urgent|high|normal|low>] [--clear-note] [--note <text>]")?;
            let task = channels::update_queue_task(
                root,
                channels::QueueTaskUpdateRequest {
                    message_key: message_key.to_string(),
                    priority: find_flag_value(args, "--priority").map(ToOwned::to_owned),
                    route_status: Some("pending".to_string()),
                    status_note: find_flag_value(args, "--note").map(ToOwned::to_owned),
                    clear_note: args.iter().any(|arg| arg == "--clear-note")
                        || find_flag_value(args, "--note").is_none(),
                    ..Default::default()
                },
            )?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "complete" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue complete --message-key <key> [--note <text>]")?;
            let task = channels::update_queue_task(
                root,
                channels::QueueTaskUpdateRequest {
                    message_key: message_key.to_string(),
                    route_status: Some("handled".to_string()),
                    status_note: find_flag_value(args, "--note").map(ToOwned::to_owned),
                    clear_note: find_flag_value(args, "--note").is_none(),
                    ..Default::default()
                },
            )?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "fail" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue fail --message-key <key> --reason <text>")?;
            let reason = required_flag_value(args, "--reason")
                .context("usage: ctox queue fail --message-key <key> --reason <text>")?;
            let task = channels::update_queue_task(
                root,
                channels::QueueTaskUpdateRequest {
                    message_key: message_key.to_string(),
                    route_status: Some("failed".to_string()),
                    status_note: Some(reason.to_string()),
                    ..Default::default()
                },
            )?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "cancel" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue cancel --message-key <key> [--reason <text>]")?;
            let task = channels::update_queue_task(
                root,
                channels::QueueTaskUpdateRequest {
                    message_key: message_key.to_string(),
                    route_status: Some("cancelled".to_string()),
                    status_note: find_flag_value(args, "--reason").map(ToOwned::to_owned),
                    clear_note: find_flag_value(args, "--reason").is_none(),
                    ..Default::default()
                },
            )?;
            print_json(&json!({"ok": true, "task": task}))
        }
        "spill" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue spill --message-key <key> [--ticket-system <name>] [--reason <text>] [--skill <name>] [--publish]")?;
            let bridge = spill_queue_task_to_ticket(
                root,
                message_key,
                find_flag_value(args, "--ticket-system").unwrap_or(DEFAULT_TICKET_SYSTEM),
                find_flag_value(args, "--reason"),
                find_flag_value(args, "--skill"),
                args.iter().any(|arg| arg == "--publish"),
            )?;
            print_json(&json!({"ok": true, "bridge": bridge}))
        }
        "spill-candidates" => {
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_LIST_LIMIT);
            let candidates = list_queue_spill_candidates(root, limit)?;
            print_json(&json!({"ok": true, "count": candidates.len(), "candidates": candidates}))
        }
        "spills" => {
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_LIST_LIMIT);
            let bridges = list_queue_ticket_bridges(root, find_flag_value(args, "--state"), limit)?;
            print_json(&json!({"ok": true, "count": bridges.len(), "spills": bridges}))
        }
        "restore" => {
            let message_key = required_flag_value(args, "--message-key")
                .context("usage: ctox queue restore --message-key <key> [--priority <urgent|high|normal|low>] [--note <text>]")?;
            let bridge = restore_spilled_queue_task(
                root,
                message_key,
                find_flag_value(args, "--priority"),
                find_flag_value(args, "--note"),
            )?;
            print_json(&json!({"ok": true, "bridge": bridge}))
        }
        _ => anyhow::bail!(
            "usage:\n  ctox queue add --title <label> --prompt <text> [--thread-key <key>] [--workspace-root <path>] [--skill <name>] [--priority <urgent|high|normal|low>] [--parent-message-key <key>]\n  ctox queue list [--status <pending|leased|blocked|failed|handled|cancelled>]... [--limit <n>]\n  ctox queue show --message-key <key>\n  ctox queue edit --message-key <key> [--title <label>] [--prompt <text>] [--thread-key <key>] [--workspace-root <path>] [--clear-workspace-root] [--skill <name>] [--clear-skill] [--priority <urgent|high|normal|low>]\n  ctox queue reprioritize --message-key <key> --priority <urgent|high|normal|low>\n  ctox queue block --message-key <key> --reason <text>\n  ctox queue release --message-key <key> [--priority <urgent|high|normal|low>] [--clear-note] [--note <text>]\n  ctox queue complete --message-key <key> [--note <text>]\n  ctox queue fail --message-key <key> --reason <text>\n  ctox queue cancel --message-key <key> [--reason <text>]\n  ctox queue spill --message-key <key> [--ticket-system <name>] [--reason <text>] [--skill <name>] [--publish]\n  ctox queue spill-candidates [--limit <n>]\n  ctox queue spills [--state <spilled|restored>] [--limit <n>]\n  ctox queue restore --message-key <key> [--priority <urgent|high|normal|low>] [--note <text>]"
        ),
    }
}

fn spill_queue_task_to_ticket(
    root: &Path,
    message_key: &str,
    ticket_system: &str,
    reason: Option<&str>,
    explicit_skill: Option<&str>,
    publish: bool,
) -> Result<QueueTicketBridgeView> {
    let task = channels::load_queue_task(root, message_key)?.context("queue task not found")?;
    let existing = load_queue_ticket_bridge(root, message_key)?;
    if let Some(existing) = existing.filter(|bridge| bridge.bridge_state == "spilled") {
        let ticket = tickets::load_ticket_self_work_item(root, &existing.work_id)?
            .context("bridged ticket self-work item missing")?;
        return Ok(QueueTicketBridgeView {
            message_key: existing.message_key,
            work_id: existing.work_id,
            ticket_system: existing.ticket_system,
            bridge_state: existing.bridge_state,
            spilled_at: existing.spilled_at,
            restored_at: existing.restored_at,
            task,
            ticket,
        });
    }

    let effective_skill = explicit_skill
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .or(task.suggested_skill.clone())
        .or_else(|| Some("queue-orchestrator".to_string()));
    let ticket = tickets::put_ticket_self_work_item(
        root,
        tickets::TicketSelfWorkUpsertInput {
            source_system: ticket_system.trim().to_string(),
            kind: "queue-overflow".to_string(),
            title: format!("Queue spill: {}", task.title),
            body_text: render_queue_spill_body(&task, reason),
            state: "spilled".to_string(),
            metadata: json!({
                "skill": effective_skill,
                "dedupe_key": task.message_key,
                "bridge_kind": "queue_spillover",
                "queue_message_key": task.message_key,
                "queue_thread_key": task.thread_key,
                "queue_priority": task.priority,
                "queue_workspace_root": task.workspace_root,
                "queue_parent_message_key": task.parent_message_key,
                "queue_task_title": task.title,
                "queue_prompt": task.prompt,
                "reason": reason.map(str::trim).filter(|value| !value.is_empty()),
            }),
        },
        publish,
    )?;
    let note = match reason.map(str::trim).filter(|value| !value.is_empty()) {
        Some(reason) => format!("spilled to ticket {}: {reason}", ticket.work_id),
        None => format!("spilled to ticket {}", ticket.work_id),
    };
    let task = channels::update_queue_task(
        root,
        channels::QueueTaskUpdateRequest {
            message_key: task.message_key.clone(),
            route_status: Some("blocked".to_string()),
            status_note: Some(note),
            ..Default::default()
        },
    )?;
    let bridge = upsert_queue_ticket_bridge(
        root,
        QueueTicketBridgeRecord {
            message_key: task.message_key.clone(),
            work_id: ticket.work_id.clone(),
            ticket_system: ticket.source_system.clone(),
            bridge_state: "spilled".to_string(),
            spilled_at: now_iso_string(),
            restored_at: None,
        },
    )?;
    let _ = ensure_spill_restore_follow_up(root, &task, &ticket.work_id, reason)?;
    Ok(QueueTicketBridgeView {
        message_key: bridge.message_key,
        work_id: bridge.work_id,
        ticket_system: bridge.ticket_system,
        bridge_state: bridge.bridge_state,
        spilled_at: bridge.spilled_at,
        restored_at: bridge.restored_at,
        task,
        ticket,
    })
}

fn restore_spilled_queue_task(
    root: &Path,
    message_key: &str,
    priority: Option<&str>,
    note: Option<&str>,
) -> Result<QueueTicketBridgeView> {
    let bridge = load_queue_ticket_bridge(root, message_key)?
        .context("queue task is not currently linked to a spilled ticket")?;
    anyhow::ensure!(
        bridge.bridge_state == "spilled",
        "queue task is not in spilled state"
    );
    let current_task =
        channels::load_queue_task(root, message_key)?.context("queue task not found")?;
    let restored_title = restored_queue_follow_up_title(&current_task.title);
    let restored_note = note
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| {
            format!(
                "restored from ticket {}; held as the single open follow-up for the next turn",
                bridge.work_id
            )
        });
    let _ = channels::update_queue_task(
        root,
        channels::QueueTaskUpdateRequest {
            message_key: message_key.to_string(),
            title: Some(restored_title),
            priority: priority.map(ToOwned::to_owned),
            status_note: Some(restored_note),
            ..Default::default()
        },
    )?;
    complete_spill_restore_follow_ups(root, message_key)?;
    let task = channels::lease_queue_task(root, message_key, SPILL_RESTORE_LEASE_OWNER)?;
    let ticket = tickets::set_ticket_self_work_state(root, &bridge.work_id, "restored")?;
    let bridge = upsert_queue_ticket_bridge(
        root,
        QueueTicketBridgeRecord {
            message_key: bridge.message_key.clone(),
            work_id: bridge.work_id.clone(),
            ticket_system: bridge.ticket_system.clone(),
            bridge_state: "restored".to_string(),
            spilled_at: bridge.spilled_at,
            restored_at: Some(now_iso_string()),
        },
    )?;
    Ok(QueueTicketBridgeView {
        message_key: bridge.message_key,
        work_id: bridge.work_id,
        ticket_system: bridge.ticket_system,
        bridge_state: bridge.bridge_state,
        spilled_at: bridge.spilled_at,
        restored_at: bridge.restored_at,
        task,
        ticket,
    })
}

#[derive(Debug, Clone)]
struct QueueTicketBridgeRecord {
    message_key: String,
    work_id: String,
    ticket_system: String,
    bridge_state: String,
    spilled_at: String,
    restored_at: Option<String>,
}

fn spill_restore_follow_up_title(current_title: &str) -> String {
    let trimmed = current_title.trim();
    if trimmed.is_empty() {
        format!("{SPILL_RESTORE_TITLE_PREFIX}spilled queue task")
    } else {
        format!("{SPILL_RESTORE_TITLE_PREFIX}{trimmed}")
    }
}

fn render_spill_restore_follow_up_prompt(
    task: &channels::QueueTaskView,
    work_id: &str,
    reason: Option<&str>,
) -> String {
    let mut lines = vec![
        "Restore the spilled queue task from internal ticket self-work when the queue is ready."
            .to_string(),
        format!("Original queue task: {}", task.title.trim()),
        format!("Queue message key: {}", task.message_key),
        format!("Ticket self-work id: {work_id}"),
        "Required actions:".to_string(),
        "- confirm the spill is still the right choice".to_string(),
        "- restore the original queue task with `ctox queue restore --message-key <key>`"
            .to_string(),
        "- keep exactly one open CTOX follow-up after restore".to_string(),
    ];
    if let Some(reason) = reason.map(str::trim).filter(|value| !value.is_empty()) {
        lines.push(format!("Spill reason: {reason}"));
    }
    lines.join("\n")
}

fn ensure_spill_restore_follow_up(
    root: &Path,
    task: &channels::QueueTaskView,
    work_id: &str,
    reason: Option<&str>,
) -> Result<channels::QueueTaskView> {
    let existing = channels::list_queue_tasks(root, &[], 256)?
        .into_iter()
        .find(|candidate| {
            candidate.parent_message_key.as_deref() == Some(task.message_key.as_str())
                && candidate
                    .title
                    .to_ascii_lowercase()
                    .starts_with(SPILL_RESTORE_TITLE_PREFIX)
        });
    let follow_up = if let Some(existing) = existing {
        channels::update_queue_task(
            root,
            channels::QueueTaskUpdateRequest {
                message_key: existing.message_key,
                title: Some(spill_restore_follow_up_title(&task.title)),
                prompt: Some(render_spill_restore_follow_up_prompt(task, work_id, reason)),
                priority: Some("high".to_string()),
                suggested_skill: task.suggested_skill.clone(),
                ..Default::default()
            },
        )?
    } else {
        channels::create_queue_task(
            root,
            channels::QueueTaskCreateRequest {
                title: spill_restore_follow_up_title(&task.title),
                prompt: render_spill_restore_follow_up_prompt(task, work_id, reason),
                thread_key: task.thread_key.clone(),
                workspace_root: task.workspace_root.clone(),
                priority: "high".to_string(),
                suggested_skill: task.suggested_skill.clone(),
                parent_message_key: Some(task.message_key.clone()),
            },
        )?
    };
    channels::lease_queue_task(root, &follow_up.message_key, SPILL_RESTORE_LEASE_OWNER)
}

fn complete_spill_restore_follow_ups(root: &Path, parent_message_key: &str) -> Result<()> {
    for follow_up in channels::list_queue_tasks(root, &[], 256)?
        .into_iter()
        .filter(|task| {
            task.parent_message_key.as_deref() == Some(parent_message_key)
                && task
                    .title
                    .to_ascii_lowercase()
                    .starts_with(SPILL_RESTORE_TITLE_PREFIX)
                && task.route_status != "handled"
        })
    {
        let _ = channels::update_queue_task(
            root,
            channels::QueueTaskUpdateRequest {
                message_key: follow_up.message_key,
                route_status: Some("handled".to_string()),
                status_note: Some("superseded by the restored original queue task".to_string()),
                ..Default::default()
            },
        )?;
    }
    Ok(())
}

fn restored_queue_follow_up_title(current_title: &str) -> String {
    let trimmed = current_title.trim();
    let normalized = trimmed.to_ascii_lowercase();
    if normalized.starts_with("restored queue") || normalized.starts_with("restored follow-up") {
        return trimmed.to_string();
    }
    if trimmed.is_empty() {
        "restored queue rehydrate follow-up".to_string()
    } else {
        format!("restored queue rehydrate follow-up: {trimmed}")
    }
}

fn render_queue_spill_body(task: &channels::QueueTaskView, reason: Option<&str>) -> String {
    let mut lines = vec![
        "This task was spilled out of the CTOX queue into the internal ticket system.".to_string(),
        format!("Queue task: {}", task.title),
        format!("Queue message key: {}", task.message_key),
        format!("Thread: {}", task.thread_key),
        format!("Priority: {}", task.priority),
    ];
    if let Some(workspace_root) = task.workspace_root.as_deref() {
        lines.push(format!("Workspace: {}", workspace_root));
    }
    if let Some(skill) = task.suggested_skill.as_deref() {
        lines.push(format!("Suggested skill: {}", skill));
    }
    if let Some(parent) = task.parent_message_key.as_deref() {
        lines.push(format!("Parent queue message: {}", parent));
    }
    if let Some(reason) = reason.map(str::trim).filter(|value| !value.is_empty()) {
        lines.push(String::new());
        lines.push(format!("Spill reason: {}", reason));
    }
    lines.push(String::new());
    lines.push("Original prompt:".to_string());
    lines.push(task.prompt.clone());
    lines.join("\n")
}

fn queue_bridge_db_path(root: &Path) -> std::path::PathBuf {
    crate::paths::mission_db(root)
}

fn open_queue_bridge_db(root: &Path) -> Result<Connection> {
    let path = queue_bridge_db_path(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let conn = Connection::open(&path)
        .with_context(|| format!("failed to open queue bridge db {}", path.display()))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;
    ensure_queue_bridge_schema(&conn)?;
    Ok(conn)
}

fn ensure_queue_bridge_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS queue_ticket_spills (
            message_key TEXT PRIMARY KEY,
            work_id TEXT NOT NULL,
            ticket_system TEXT NOT NULL,
            bridge_state TEXT NOT NULL,
            spilled_at TEXT NOT NULL,
            restored_at TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_queue_ticket_spills_work
            ON queue_ticket_spills(work_id, updated_at DESC);
        "#,
    )?;
    Ok(())
}

fn upsert_queue_ticket_bridge(
    root: &Path,
    record: QueueTicketBridgeRecord,
) -> Result<QueueTicketBridgeRecord> {
    let conn = open_queue_bridge_db(root)?;
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO queue_ticket_spills (
            message_key, work_id, ticket_system, bridge_state, spilled_at, restored_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
        ON CONFLICT(message_key) DO UPDATE SET
            work_id=excluded.work_id,
            ticket_system=excluded.ticket_system,
            bridge_state=excluded.bridge_state,
            spilled_at=excluded.spilled_at,
            restored_at=excluded.restored_at,
            updated_at=excluded.updated_at
        "#,
        params![
            record.message_key,
            record.work_id,
            record.ticket_system,
            record.bridge_state,
            record.spilled_at,
            record.restored_at,
            now,
        ],
    )?;
    load_queue_ticket_bridge(root, &record.message_key)?
        .context("queue ticket bridge missing after write")
}

fn load_queue_ticket_bridge(
    root: &Path,
    message_key: &str,
) -> Result<Option<QueueTicketBridgeRecord>> {
    let conn = open_queue_bridge_db(root)?;
    conn.query_row(
        r#"
        SELECT message_key, work_id, ticket_system, bridge_state, spilled_at, restored_at
        FROM queue_ticket_spills
        WHERE message_key = ?1
        LIMIT 1
        "#,
        params![message_key],
        |row| {
            Ok(QueueTicketBridgeRecord {
                message_key: row.get(0)?,
                work_id: row.get(1)?,
                ticket_system: row.get(2)?,
                bridge_state: row.get(3)?,
                spilled_at: row.get(4)?,
                restored_at: row.get(5)?,
            })
        },
    )
    .optional()
    .map_err(anyhow::Error::from)
}

fn list_queue_ticket_bridges(
    root: &Path,
    state: Option<&str>,
    limit: usize,
) -> Result<Vec<QueueTicketBridgeListItem>> {
    let conn = open_queue_bridge_db(root)?;
    let mut statement = conn.prepare(
        r#"
        SELECT message_key, work_id, ticket_system, bridge_state, spilled_at, restored_at
        FROM queue_ticket_spills
        WHERE (?1 IS NULL OR bridge_state = ?1)
        ORDER BY updated_at DESC, spilled_at DESC
        LIMIT ?2
        "#,
    )?;
    let rows = statement.query_map(params![state, limit as i64], |row| {
        Ok(QueueTicketBridgeRecord {
            message_key: row.get(0)?,
            work_id: row.get(1)?,
            ticket_system: row.get(2)?,
            bridge_state: row.get(3)?,
            spilled_at: row.get(4)?,
            restored_at: row.get(5)?,
        })
    })?;
    let mut items = Vec::new();
    for row in rows {
        let bridge = row?;
        let task = channels::load_queue_task(root, &bridge.message_key)?;
        let ticket = tickets::load_ticket_self_work_item(root, &bridge.work_id)?;
        items.push(QueueTicketBridgeListItem {
            message_key: bridge.message_key,
            work_id: bridge.work_id,
            ticket_system: bridge.ticket_system,
            bridge_state: bridge.bridge_state,
            spilled_at: bridge.spilled_at,
            restored_at: bridge.restored_at,
            task,
            ticket,
        });
    }
    Ok(items)
}

fn list_queue_spill_candidates(root: &Path, limit: usize) -> Result<Vec<QueueSpillCandidateView>> {
    let tasks = channels::list_queue_tasks(
        root,
        &["pending".to_string(), "blocked".to_string()],
        10_000,
    )?;
    let mut candidates = Vec::new();
    for task in tasks {
        if let Some(candidate) = score_queue_spill_candidate(root, task)? {
            candidates.push(candidate);
        }
    }
    candidates.sort_by(|left, right| {
        right
            .candidate_score
            .cmp(&left.candidate_score)
            .then_with(|| left.priority.cmp(&right.priority))
            .then_with(|| left.title.cmp(&right.title))
    });
    candidates.truncate(limit);
    Ok(candidates)
}

fn score_queue_spill_candidate(
    root: &Path,
    task: channels::QueueTaskView,
) -> Result<Option<QueueSpillCandidateView>> {
    if let Some(existing) = load_queue_ticket_bridge(root, &task.message_key)? {
        if existing.bridge_state == "spilled" {
            return Ok(None);
        }
    }
    if matches!(task.route_status.as_str(), "handled" | "cancelled") {
        return Ok(None);
    }

    let mut score = 0i64;
    let mut reasons = Vec::new();

    match task.priority.as_str() {
        "low" => {
            score += 4;
            reasons.push("priority is low, so it can leave the hot queue first".to_string());
        }
        "normal" => {
            score += 2;
            reasons.push(
                "priority is normal and can be deferred if queue pressure is high".to_string(),
            );
        }
        "high" => {
            score -= 1;
            reasons.push(
                "priority is high, so spill only if higher-risk work must stay hot".to_string(),
            );
        }
        "urgent" => {
            reasons.push(
                "priority is urgent, so it should normally stay in the hot queue".to_string(),
            );
            return Ok(None);
        }
        _ => {
            score += 1;
            reasons.push("priority is not classified as urgent".to_string());
        }
    }

    match task.route_status.as_str() {
        "blocked" => {
            score += 5;
            reasons.push("task is already blocked, so moving it into internal ticket tracking reduces queue pressure without losing it".to_string());
        }
        "pending" => {
            score += 1;
        }
        _ => {}
    }

    if task.workspace_root.is_some() {
        score += 1;
        reasons.push(
            "workspace context is already attached, which makes later restoration safer"
                .to_string(),
        );
    }
    if task.suggested_skill.is_some() {
        score += 1;
        reasons.push(
            "task already names a suggested skill, so it can re-enter the loop cleanly later"
                .to_string(),
        );
    }
    if task.parent_message_key.is_some() {
        score -= 2;
        reasons.push(
            "task has a parent queue message, so spilling it may hide active continuity"
                .to_string(),
        );
    }

    if score <= 0 {
        return Ok(None);
    }

    let recommendation = if task.route_status == "blocked" {
        "strong spill candidate".to_string()
    } else if score >= 5 {
        "good spill candidate".to_string()
    } else {
        "spill only if pressure remains high".to_string()
    };

    Ok(Some(QueueSpillCandidateView {
        message_key: task.message_key,
        priority: task.priority,
        route_status: task.route_status,
        title: task.title,
        thread_key: task.thread_key,
        suggested_skill: task.suggested_skill,
        workspace_root: task.workspace_root,
        candidate_score: score,
        recommendation,
        reasons,
    }))
}

fn now_iso_string() -> String {
    chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true)
}

fn ensure_edit_requested(args: &[String], value_flags: &[&str], bool_flags: &[&str]) -> Result<()> {
    let has_value_change = value_flags
        .iter()
        .any(|flag| find_flag_value(args, flag).is_some());
    let has_bool_change = bool_flags
        .iter()
        .any(|flag| args.iter().any(|arg| arg == flag));
    if has_value_change || has_bool_change {
        return Ok(());
    }
    anyhow::bail!("queue edit requires at least one field change")
}

fn default_thread_key(title: &str) -> String {
    let slug = title
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .split('-')
        .filter(|part| !part.is_empty())
        .take(6)
        .collect::<Vec<_>>()
        .join("-");
    let digest = stable_digest(title);
    if slug.is_empty() {
        format!("queue/task-{digest}")
    } else {
        format!("queue/{slug}-{digest}")
    }
}

fn stable_digest(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let hex = format!("{digest:x}");
    hex[..12].to_string()
}

fn required_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    find_flag_value(args, flag)
}

fn find_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let index = args.iter().position(|arg| arg == flag)?;
    args.get(index + 1).map(String::as_str)
}

fn collect_flag_values(args: &[String], flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == flag {
            if let Some(value) = args.get(index + 1) {
                values.push(value.clone());
                index += 2;
                continue;
            }
        }
        index += 1;
    }
    values
}

fn print_json(value: &serde_json::Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(label: &str) -> std::path::PathBuf {
        let mut path = std::env::temp_dir();
        path.push(format!("ctox-queue-test-{}-{}", label, std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        path
    }

    #[test]
    fn queue_task_can_spill_to_internal_ticket_and_restore() -> Result<()> {
        let root = temp_root("spill-restore");
        std::fs::create_dir_all(&root)?;

        let task = channels::create_queue_task(
            &root,
            channels::QueueTaskCreateRequest {
                title: "Investigate monitoring drift".to_string(),
                prompt: "Inspect Prometheus drift and report likely root cause.".to_string(),
                thread_key: "queue/monitoring-drift".to_string(),
                workspace_root: Some("/tmp/monitoring".to_string()),
                priority: "high".to_string(),
                suggested_skill: Some("reliability-ops".to_string()),
                parent_message_key: None,
            },
        )?;

        let spilled = spill_queue_task_to_ticket(
            &root,
            &task.message_key,
            DEFAULT_TICKET_SYSTEM,
            Some("queue pressure exceeded safe working set"),
            None,
            false,
        )?;
        assert_eq!(spilled.bridge_state, "spilled");
        assert_eq!(spilled.task.route_status, "blocked");
        let open_after_spill =
            channels::list_queue_tasks(&root, &["pending".to_string(), "leased".to_string()], 10)?;
        assert_eq!(open_after_spill.len(), 1);
        assert_eq!(open_after_spill[0].route_status, "leased");
        assert_eq!(
            open_after_spill[0].lease_owner.as_deref(),
            Some(SPILL_RESTORE_LEASE_OWNER)
        );
        assert!(open_after_spill[0].title.starts_with("spill restore:"));
        assert_eq!(spilled.ticket.kind, "queue-overflow");
        assert_eq!(
            spilled.ticket.suggested_skill.as_deref(),
            Some("reliability-ops")
        );
        assert_eq!(
            spilled
                .ticket
                .metadata
                .get("queue_message_key")
                .and_then(serde_json::Value::as_str),
            Some(task.message_key.as_str())
        );

        let restored = restore_spilled_queue_task(
            &root,
            &task.message_key,
            Some("urgent"),
            Some("resume after ticket review"),
        )?;
        assert_eq!(restored.bridge_state, "restored");
        assert_eq!(restored.task.route_status, "leased");
        assert_eq!(
            restored.task.lease_owner.as_deref(),
            Some(SPILL_RESTORE_LEASE_OWNER)
        );
        assert!(restored
            .task
            .title
            .starts_with("restored queue rehydrate follow-up:"));
        assert_eq!(restored.task.priority, "urgent");
        assert_eq!(restored.ticket.state, "restored");
        let open_after_restore =
            channels::list_queue_tasks(&root, &["pending".to_string(), "leased".to_string()], 10)?;
        assert_eq!(open_after_restore.len(), 1);
        assert_eq!(open_after_restore[0].message_key, restored.task.message_key);

        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn spill_candidates_rank_blocked_and_lower_priority_tasks_first() -> Result<()> {
        let root = temp_root("spill-candidates");
        std::fs::create_dir_all(&root)?;

        let blocked = channels::create_queue_task(
            &root,
            channels::QueueTaskCreateRequest {
                title: "Blocked low-priority audit".to_string(),
                prompt: "Review old audit findings.".to_string(),
                thread_key: "queue/audit".to_string(),
                workspace_root: Some("/tmp/audit".to_string()),
                priority: "low".to_string(),
                suggested_skill: Some("audit-review".to_string()),
                parent_message_key: None,
            },
        )?;
        let _ = channels::update_queue_task(
            &root,
            channels::QueueTaskUpdateRequest {
                message_key: blocked.message_key.clone(),
                route_status: Some("blocked".to_string()),
                status_note: Some("waiting for quieter window".to_string()),
                ..Default::default()
            },
        )?;

        let urgent = channels::create_queue_task(
            &root,
            channels::QueueTaskCreateRequest {
                title: "Urgent prod incident".to_string(),
                prompt: "Handle production incident.".to_string(),
                thread_key: "queue/incident".to_string(),
                workspace_root: None,
                priority: "urgent".to_string(),
                suggested_skill: Some("incident-response".to_string()),
                parent_message_key: None,
            },
        )?;

        let candidates = list_queue_spill_candidates(&root, 10)?;
        assert_eq!(
            candidates.first().map(|item| item.message_key.as_str()),
            Some(blocked.message_key.as_str())
        );
        assert!(!candidates
            .iter()
            .any(|item| item.message_key == urgent.message_key));
        assert!(candidates
            .first()
            .map(|item| item
                .reasons
                .iter()
                .any(|reason| reason.contains("already blocked")))
            .unwrap_or(false));

        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }

    #[test]
    fn spills_list_returns_joined_queue_and_ticket_state() -> Result<()> {
        let root = temp_root("spills-list");
        std::fs::create_dir_all(&root)?;

        let task = channels::create_queue_task(
            &root,
            channels::QueueTaskCreateRequest {
                title: "Deferred documentation review".to_string(),
                prompt: "Review documentation backlog.".to_string(),
                thread_key: "queue/docs".to_string(),
                workspace_root: None,
                priority: "normal".to_string(),
                suggested_skill: Some("docs-review".to_string()),
                parent_message_key: None,
            },
        )?;
        let bridge = spill_queue_task_to_ticket(
            &root,
            &task.message_key,
            DEFAULT_TICKET_SYSTEM,
            None,
            None,
            false,
        )?;

        let spills = list_queue_ticket_bridges(&root, Some("spilled"), 10)?;
        assert_eq!(spills.len(), 1);
        assert_eq!(spills[0].message_key, task.message_key);
        assert_eq!(spills[0].work_id, bridge.work_id);
        assert_eq!(
            spills[0].ticket.as_ref().map(|item| item.kind.as_str()),
            Some("queue-overflow")
        );
        assert_eq!(
            spills[0]
                .task
                .as_ref()
                .map(|item| item.route_status.as_str()),
            Some("blocked")
        );

        let _ = std::fs::remove_dir_all(&root);
        Ok(())
    }
}

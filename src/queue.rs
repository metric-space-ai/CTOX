use anyhow::Context;
use anyhow::Result;
use serde_json::json;
use sha2::Digest;
use sha2::Sha256;
use std::path::Path;

use crate::channels;

const DEFAULT_LIST_LIMIT: usize = 20;

pub fn handle_queue_command(root: &Path, args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("");
    match command {
        "add" => {
            let title = required_flag_value(args, "--title")
                .context("usage: ctox queue add --title <label> --prompt <text> [--thread-key <key>] [--skill <name>] [--priority <urgent|high|normal|low>] [--parent-message-key <key>]")?;
            let prompt = required_flag_value(args, "--prompt")
                .context("usage: ctox queue add --title <label> --prompt <text> [--thread-key <key>] [--skill <name>] [--priority <urgent|high|normal|low>] [--parent-message-key <key>]")?;
            let thread_key = find_flag_value(args, "--thread-key")
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| default_thread_key(title));
            let task = channels::create_queue_task(
                root,
                channels::QueueTaskCreateRequest {
                    title: title.to_string(),
                    prompt: prompt.to_string(),
                    thread_key,
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
                .context("usage: ctox queue edit --message-key <key> [--title <label>] [--prompt <text>] [--thread-key <key>] [--skill <name>] [--clear-skill] [--priority <urgent|high|normal|low>]")?;
            ensure_edit_requested(args, &["--title", "--prompt", "--thread-key", "--skill", "--priority"], &["--clear-skill"])?;
            let task = channels::update_queue_task(
                root,
                channels::QueueTaskUpdateRequest {
                    message_key: message_key.to_string(),
                    title: find_flag_value(args, "--title").map(ToOwned::to_owned),
                    prompt: find_flag_value(args, "--prompt").map(ToOwned::to_owned),
                    thread_key: find_flag_value(args, "--thread-key").map(ToOwned::to_owned),
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
        _ => anyhow::bail!(
            "usage:\n  ctox queue add --title <label> --prompt <text> [--thread-key <key>] [--skill <name>] [--priority <urgent|high|normal|low>] [--parent-message-key <key>]\n  ctox queue list [--status <pending|leased|blocked|failed|handled|cancelled>]... [--limit <n>]\n  ctox queue show --message-key <key>\n  ctox queue edit --message-key <key> [--title <label>] [--prompt <text>] [--thread-key <key>] [--skill <name>] [--clear-skill] [--priority <urgent|high|normal|low>]\n  ctox queue reprioritize --message-key <key> --priority <urgent|high|normal|low>\n  ctox queue block --message-key <key> --reason <text>\n  ctox queue release --message-key <key> [--priority <urgent|high|normal|low>] [--clear-note] [--note <text>]\n  ctox queue complete --message-key <key> [--note <text>]\n  ctox queue fail --message-key <key> --reason <text>\n  ctox queue cancel --message-key <key> [--reason <text>]"
        ),
    }
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

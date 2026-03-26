use anyhow::Context;
use anyhow::Result;
use chrono::DateTime;
use chrono::Duration;
use chrono::Utc;
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use rusqlite::Transaction;
use serde::Serialize;
use serde_json::json;
use serde_json::Value;
use sha2::Digest;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::runtime_config;

const DEFAULT_DB_RELATIVE_PATH: &str = "runtime/cto_agent.db";
const DEFAULT_TAKE_LIMIT: usize = 10;
const QUEUE_CHANNEL_NAME: &str = "queue";
const QUEUE_ACCOUNT_KEY: &str = "queue:system";
const QUEUE_ACCOUNT_ADDRESS: &str = "ctox queue";
const QUEUE_PROVIDER: &str = "system";
const QUEUE_SENDER_DISPLAY: &str = "CTOX queue";
const QUEUE_SENDER_ADDRESS: &str = "queue:system";

#[derive(Debug, Clone, Serialize)]
pub struct QueueTaskView {
    pub message_key: String,
    pub thread_key: String,
    pub title: String,
    pub prompt: String,
    pub priority: String,
    pub suggested_skill: Option<String>,
    pub parent_message_key: Option<String>,
    pub route_status: String,
    pub status_note: Option<String>,
    pub lease_owner: Option<String>,
    pub leased_at: Option<String>,
    pub acked_at: Option<String>,
    pub created_at: String,
    pub sort_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone)]
pub struct QueueTaskCreateRequest {
    pub title: String,
    pub prompt: String,
    pub thread_key: String,
    pub priority: String,
    pub suggested_skill: Option<String>,
    pub parent_message_key: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct QueueTaskUpdateRequest {
    pub message_key: String,
    pub title: Option<String>,
    pub prompt: Option<String>,
    pub thread_key: Option<String>,
    pub priority: Option<String>,
    pub suggested_skill: Option<String>,
    pub clear_skill: bool,
    pub route_status: Option<String>,
    pub status_note: Option<String>,
    pub clear_note: bool,
}

pub struct OwnerPromptContext {
    pub owner_name: String,
    pub owner_email_address: Option<String>,
    pub allowed_email_domain: Option<String>,
    pub admin_email_policies: Vec<String>,
    pub channels: Vec<String>,
    pub preferred_channel: Option<String>,
}

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub struct EmailSenderPolicy {
    pub normalized_email: String,
    pub role: String,
    pub allowed: bool,
    pub allow_admin_actions: bool,
    pub allow_sudo_actions: bool,
    pub secrets_via_email_allowed: bool,
    pub allowed_email_domain: Option<String>,
    pub block_reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AdminEmailPolicy {
    email: String,
    can_sudo: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct CommunicationFeedItem {
    pub message_key: String,
    pub channel: String,
    pub direction: String,
    pub sender_display: String,
    pub sender_address: String,
    pub subject: String,
    pub preview: String,
    pub thread_key: String,
    pub route_status: String,
    pub external_created_at: String,
}

pub fn sync_prompt_identity(root: &Path, settings: &BTreeMap<String, String>) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    if let Some(owner_name) = settings
        .get("CTOX_OWNER_NAME")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        upsert_owner_profile(&mut conn, owner_name)?;
    }
    sync_identity_profiles(&mut conn, settings)?;

    if let Some(email_address) = settings
        .get("CTO_EMAIL_ADDRESS")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        let provider = settings
            .get("CTO_EMAIL_PROVIDER")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .unwrap_or("imap");
        let profile_json = json!({
            "imapHost": settings.get("CTO_EMAIL_IMAP_HOST").map(|value| value.trim()).unwrap_or(""),
            "imapPort": settings.get("CTO_EMAIL_IMAP_PORT").map(|value| value.trim()).unwrap_or(""),
            "smtpHost": settings.get("CTO_EMAIL_SMTP_HOST").map(|value| value.trim()).unwrap_or(""),
            "smtpPort": settings.get("CTO_EMAIL_SMTP_PORT").map(|value| value.trim()).unwrap_or(""),
            "graphUser": settings.get("CTO_EMAIL_GRAPH_USER").map(|value| value.trim()).unwrap_or(""),
            "ewsUrl": settings.get("CTO_EMAIL_EWS_URL").map(|value| value.trim()).unwrap_or(""),
            "ewsAuthType": settings.get("CTO_EMAIL_EWS_AUTH_TYPE").map(|value| value.trim()).unwrap_or(""),
            "ewsUsername": settings.get("CTO_EMAIL_EWS_USERNAME").map(|value| value.trim()).unwrap_or(""),
        });
        ensure_account(
            &mut conn,
            &format!("email:{email_address}"),
            "email",
            email_address,
            provider,
            profile_json,
        )?;
    }

    if let Some(jami_account_id) = settings
        .get("CTO_JAMI_ACCOUNT_ID")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
    {
        let profile_name = settings
            .get("CTO_JAMI_PROFILE_NAME")
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
            .unwrap_or(jami_account_id);
        let profile_json = json!({
            "accountId": jami_account_id,
            "profileName": profile_name,
            "inboxDir": settings.get("CTO_JAMI_INBOX_DIR").map(|value| value.trim()).unwrap_or(""),
            "outboxDir": settings.get("CTO_JAMI_OUTBOX_DIR").map(|value| value.trim()).unwrap_or(""),
            "archiveDir": settings.get("CTO_JAMI_ARCHIVE_DIR").map(|value| value.trim()).unwrap_or(""),
            "dbusEnvFile": settings.get("CTO_JAMI_DBUS_ENV_FILE").map(|value| value.trim()).unwrap_or(""),
        });
        ensure_account(
            &mut conn,
            &format!("jami:{jami_account_id}"),
            "jami",
            profile_name,
            "jami",
            profile_json,
        )?;
    }

    Ok(())
}

pub fn load_prompt_identity(
    root: &Path,
    settings: &BTreeMap<String, String>,
) -> Result<OwnerPromptContext> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let owner_name = load_owner_name(&conn)?
        .or_else(|| {
            settings
                .get("CTOX_OWNER_NAME")
                .map(|value| value.trim().to_string())
                .filter(|value| !value.is_empty())
        })
        .unwrap_or_else(|| "the owner".to_string());

    let mut channels = BTreeSet::new();
    channels.insert("- tui: direct local CTOX session".to_string());

    let mut stmt = conn.prepare(
        r#"
        SELECT channel, address, provider, profile_json
        FROM communication_accounts
        ORDER BY channel ASC, account_key ASC
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    for row in rows {
        let (channel, address, provider, profile_json) = row?;
        match channel.as_str() {
            "email" => {
                if !address.trim().is_empty() {
                    channels.insert(format!(
                        "- email: {} (provider: {})",
                        address.trim(),
                        provider.trim()
                    ));
                }
            }
            "jami" => {
                let parsed =
                    serde_json::from_str::<Value>(&profile_json).unwrap_or_else(|_| json!({}));
                let profile_name = parsed
                    .get("profileName")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or(address.trim());
                if !profile_name.is_empty() {
                    channels.insert(format!("- jami: {}", profile_name));
                }
            }
            "cron" | "plan" | "queue" => {}
            other => {
                if !address.trim().is_empty() {
                    channels.insert(format!("- {}: {}", other, address.trim()));
                } else {
                    channels.insert(format!("- {}", other));
                }
            }
        }
    }

    Ok(OwnerPromptContext {
        owner_name,
        owner_email_address: settings
            .get("CTOX_OWNER_EMAIL_ADDRESS")
            .map(|value| normalize_email_address(value))
            .filter(|value| !value.is_empty()),
        allowed_email_domain: normalized_allowed_email_domain(settings),
        admin_email_policies: admin_email_policy_summaries(settings),
        channels: channels.into_iter().collect(),
        preferred_channel: settings
            .get("CTOX_OWNER_PREFERRED_CHANNEL")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    })
}

pub fn classify_email_sender(
    settings: &BTreeMap<String, String>,
    sender_address: &str,
) -> EmailSenderPolicy {
    let normalized_email = normalize_email_address(sender_address);
    let owner_email = settings
        .get("CTOX_OWNER_EMAIL_ADDRESS")
        .map(|value| normalize_email_address(value))
        .filter(|value| !value.is_empty());
    let allowed_email_domain = normalized_allowed_email_domain(settings);
    let admin_policies = parse_admin_email_policies(settings);

    if normalized_email.is_empty() {
        return EmailSenderPolicy {
            normalized_email,
            role: "external".to_string(),
            allowed: false,
            allow_admin_actions: false,
            allow_sudo_actions: false,
            secrets_via_email_allowed: false,
            allowed_email_domain,
            block_reason: Some("sender email address is empty".to_string()),
        };
    }

    if owner_email.as_deref() == Some(normalized_email.as_str()) {
        return EmailSenderPolicy {
            normalized_email,
            role: "owner".to_string(),
            allowed: true,
            allow_admin_actions: true,
            allow_sudo_actions: true,
            secrets_via_email_allowed: false,
            allowed_email_domain,
            block_reason: None,
        };
    }

    if let Some(admin) = admin_policies
        .iter()
        .find(|entry| entry.email == normalized_email)
    {
        return EmailSenderPolicy {
            normalized_email,
            role: "admin".to_string(),
            allowed: true,
            allow_admin_actions: true,
            allow_sudo_actions: admin.can_sudo,
            secrets_via_email_allowed: false,
            allowed_email_domain,
            block_reason: None,
        };
    }

    if let Some(domain) = allowed_email_domain.clone() {
        if email_matches_domain(&normalized_email, &domain) {
            return EmailSenderPolicy {
                normalized_email,
                role: "domain_user".to_string(),
                allowed: true,
                allow_admin_actions: false,
                allow_sudo_actions: false,
                secrets_via_email_allowed: false,
                allowed_email_domain: Some(domain),
                block_reason: None,
            };
        }
    }

    EmailSenderPolicy {
        normalized_email,
        role: "external".to_string(),
        allowed: false,
        allow_admin_actions: false,
        allow_sudo_actions: false,
        secrets_via_email_allowed: false,
        allowed_email_domain,
        block_reason: Some("sender is outside the allowed email domain".to_string()),
    }
}

pub fn handle_channel_command(root: &Path, args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("");
    match command {
        "init" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let conn = open_channel_db(&db_path)?;
            let result = json!({
                "ok": true,
                "db_path": db_path,
                "initialized": schema_state(&conn)?,
            });
            print_json(&result)
        }
        "sync" => {
            let channel = required_flag_value(args, "--channel")?;
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let result = sync_channel(root, &db_path, channel, args)?;
            print_json(&result)
        }
        "take" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TAKE_LIMIT);
            let lease_owner = find_flag_value(args, "--lease-owner")
                .map(ToOwned::to_owned)
                .unwrap_or_else(|| "codex".to_string());
            let channel = find_flag_value(args, "--channel").map(ToOwned::to_owned);
            let mut conn = open_channel_db(&db_path)?;
            let taken = take_messages(&mut conn, channel.as_deref(), limit, &lease_owner)?;
            print_json(&json!({
                "ok": true,
                "db_path": db_path,
                "lease_owner": lease_owner,
                "count": taken.len(),
                "messages": taken,
            }))
        }
        "ack" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let status = find_flag_value(args, "--status").unwrap_or("handled");
            let message_keys = positional_after_flags(&args[1..]);
            if message_keys.is_empty() {
                anyhow::bail!(
                    "usage: ctox channel ack [--db <path>] [--status <status>] <message-key>..."
                );
            }
            let mut conn = open_channel_db(&db_path)?;
            let updated = ack_messages(&mut conn, &message_keys, status)?;
            print_json(&json!({
                "ok": true,
                "db_path": db_path,
                "updated": updated,
                "status": status,
                "message_keys": message_keys,
            }))
        }
        "send" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let request = parse_send_request(args)?;
            let result = send_message(root, &db_path, request)?;
            print_json(&result)
        }
        "test" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let channel = required_flag_value(args, "--channel")?;
            let account_key = find_flag_value(args, "--account-key").map(ToOwned::to_owned);
            let result = test_channel(root, &db_path, channel, account_key.as_deref())?;
            print_json(&result)
        }
        "ingest-tui" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let request = parse_tui_ingest_request(args)?;
            let mut conn = open_channel_db(&db_path)?;
            let stored = ingest_tui_message(&mut conn, request)?;
            print_json(&json!({
                "ok": true,
                "db_path": db_path,
                "stored": stored,
            }))
        }
        "list" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TAKE_LIMIT);
            let channel = find_flag_value(args, "--channel");
            let conn = open_channel_db(&db_path)?;
            let messages = list_messages(&conn, channel, limit)?;
            print_json(&json!({
                "ok": true,
                "db_path": db_path,
                "count": messages.len(),
                "messages": messages,
            }))
        }
        "history" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TAKE_LIMIT);
            let thread_key = required_flag_value(args, "--thread-key")?;
            let conn = open_channel_db(&db_path)?;
            let messages = list_thread_messages(&conn, thread_key, limit)?;
            print_json(&json!({
                "ok": true,
                "db_path": db_path,
                "thread_key": thread_key,
                "count": messages.len(),
                "messages": messages,
            }))
        }
        "search" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TAKE_LIMIT);
            let query = required_flag_value(args, "--query")?;
            let channel = find_flag_value(args, "--channel");
            let sender = find_flag_value(args, "--sender");
            let conn = open_channel_db(&db_path)?;
            let messages = search_messages(&conn, query, channel, sender, limit)?;
            print_json(&json!({
                "ok": true,
                "db_path": db_path,
                "query": query,
                "channel": channel,
                "sender": sender,
                "count": messages.len(),
                "messages": messages,
            }))
        }
        _ => {
            anyhow::bail!(
                "usage:\n  ctox channel init [--db <path>]\n  ctox channel sync --channel <email|jami> [--db <path>] [adapter flags]\n  ctox channel take [--db <path>] [--channel <name>] [--limit <n>] [--lease-owner <owner>]\n  ctox channel ack [--db <path>] [--status <status>] <message-key>...\n  ctox channel send --channel <tui|email|jami> --account-key <key> --thread-key <key> --body <text> [--subject <text>] [--to <addr>]... [--send-voice]\n  ctox channel test --channel <tui|email|jami> [--db <path>] [--account-key <key>]\n  ctox channel ingest-tui --account-key <key> --thread-key <key> --body <text> [--sender-display <name>] [--sender-address <addr>] [--subject <text>]\n  ctox channel list [--db <path>] [--channel <name>] [--limit <n>]\n  ctox channel history --thread-key <key> [--db <path>] [--limit <n>]\n  ctox channel search --query <text> [--db <path>] [--channel <name>] [--sender <addr>] [--limit <n>]"
            )
        }
    }
}

pub fn load_recent_communication_feed(
    root: &Path,
    limit: usize,
) -> Result<Vec<CommunicationFeedItem>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let mut statement = conn.prepare(
        r#"
        SELECT
            m.channel,
            m.direction,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.thread_key,
            COALESCE(r.route_status, 'pending'),
            m.external_created_at
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        ORDER BY m.external_created_at DESC, m.observed_at DESC
        LIMIT ?1
        "#,
    )?;
    let rows = statement.query_map(params![limit as i64], |row| {
        Ok(CommunicationFeedItem {
            message_key: String::new(),
            channel: row.get(0)?,
            direction: row.get(1)?,
            sender_display: row.get(2)?,
            sender_address: row.get(3)?,
            subject: row.get(4)?,
            preview: row.get(5)?,
            thread_key: row.get(6)?,
            route_status: row.get(7)?,
            external_created_at: row.get(8)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub fn load_thread_communication_feed(
    root: &Path,
    thread_key: &str,
    limit: usize,
) -> Result<Vec<CommunicationFeedItem>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let mut statement = conn.prepare(
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.direction,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.thread_key,
            COALESCE(r.route_status, 'pending'),
            m.external_created_at
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.thread_key = ?1
        ORDER BY m.external_created_at DESC, m.observed_at DESC
        LIMIT ?2
        "#,
    )?;
    let rows = statement.query_map(params![thread_key, limit as i64], |row| {
        Ok(CommunicationFeedItem {
            message_key: row.get(0)?,
            channel: row.get(1)?,
            direction: row.get(2)?,
            sender_display: row.get(3)?,
            sender_address: row.get(4)?,
            subject: row.get(5)?,
            preview: row.get(6)?,
            thread_key: row.get(7)?,
            route_status: row.get(8)?,
            external_created_at: row.get(9)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub fn lease_pending_inbound_messages(
    root: &Path,
    limit: usize,
    lease_owner: &str,
) -> Result<Vec<RoutedInboundMessage>> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let leased = take_messages(&mut conn, None, limit, lease_owner)?;
    Ok(leased
        .into_iter()
        .map(|item| {
            let preferred_reply_modality = item
                .metadata
                .get("preferredReplyModality")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned);
            RoutedInboundMessage {
                message_key: item.message_key,
                channel: item.channel,
                account_key: item.account_key,
                thread_key: item.thread_key,
                sender_display: item.sender_display,
                sender_address: item.sender_address,
                subject: item.subject,
                preview: item.preview,
                body_text: item.body_text,
                external_created_at: item.external_created_at,
                metadata: item.metadata,
                preferred_reply_modality,
            }
        })
        .collect())
}

pub fn ack_leased_messages(root: &Path, message_keys: &[String], status: &str) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ack_messages(&mut conn, message_keys, status)
}

pub fn create_queue_task(root: &Path, request: QueueTaskCreateRequest) -> Result<QueueTaskView> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let title = request.title.trim();
    let prompt = request.prompt.trim();
    if title.is_empty() {
        anyhow::bail!("queue task title must not be empty");
    }
    if prompt.is_empty() {
        anyhow::bail!("queue task prompt must not be empty");
    }
    let priority = canonical_queue_priority(&request.priority)?;
    let now = now_iso_string();
    let sort_at = queue_sort_at(&priority, &now)?;
    let digest = stable_digest(&format!(
        "{}:{}:{}:{}",
        title,
        prompt,
        request.thread_key.trim(),
        now
    ));
    let message_key = format!("{QUEUE_ACCOUNT_KEY}::{digest}");
    let remote_id = format!("queue-{digest}");
    let metadata = json!({
        "source": "ctox-queue",
        "priority": priority,
        "skill": request.suggested_skill.as_deref(),
        "parent_message_key": request.parent_message_key.as_deref(),
        "created_at": now,
        "sort_at": sort_at,
    });
    upsert_communication_message(
        &mut conn,
        UpsertMessage {
            message_key: &message_key,
            channel: QUEUE_CHANNEL_NAME,
            account_key: QUEUE_ACCOUNT_KEY,
            thread_key: request.thread_key.trim(),
            remote_id: &remote_id,
            direction: "inbound",
            folder_hint: "queue",
            sender_display: QUEUE_SENDER_DISPLAY,
            sender_address: QUEUE_SENDER_ADDRESS,
            recipient_addresses_json: "[]",
            cc_addresses_json: "[]",
            bcc_addresses_json: "[]",
            subject: title,
            preview: &preview_text(prompt, title),
            body_text: prompt,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "high",
            status: "received",
            seen: false,
            has_attachments: false,
            external_created_at: &sort_at,
            observed_at: &now,
            metadata_json: &serde_json::to_string(&metadata)?,
        },
    )?;
    refresh_thread(&mut conn, request.thread_key.trim())?;
    ensure_routing_rows_for_inbound(&conn)?;
    load_queue_task_from_conn(&conn, &message_key)?.context("failed to load created queue task")
}

pub fn list_queue_tasks(
    root: &Path,
    statuses: &[String],
    limit: usize,
) -> Result<Vec<QueueTaskView>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let tasks = list_queue_tasks_from_conn(&conn, limit)?;
    if statuses.is_empty() {
        return Ok(tasks);
    }
    let allowed = statuses
        .iter()
        .map(|status| status.trim().to_lowercase())
        .filter(|status| !status.is_empty())
        .collect::<Vec<_>>();
    Ok(tasks
        .into_iter()
        .filter(|task| allowed.iter().any(|status| status == &task.route_status))
        .collect())
}

pub fn load_queue_task(root: &Path, message_key: &str) -> Result<Option<QueueTaskView>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    load_queue_task_from_conn(&conn, message_key)
}

pub fn update_queue_task(root: &Path, request: QueueTaskUpdateRequest) -> Result<QueueTaskView> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let current = load_queue_message_from_conn(&conn, &request.message_key)?
        .context("queue task not found")?;
    let current_metadata = queue_metadata_object(&current.metadata);
    let title = request
        .title
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(current.subject.trim())
        .to_string();
    let prompt = request
        .prompt
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(current.body_text.trim())
        .to_string();
    let thread_key = request
        .thread_key
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .unwrap_or(current.thread_key.trim())
        .to_string();
    let priority = if let Some(priority) = request.priority.as_deref() {
        canonical_queue_priority(priority)?
    } else {
        current_queue_priority(&current)
    };
    let now = now_iso_string();
    let sort_at = queue_sort_at(&priority, &now)?;
    let mut metadata = current_metadata;
    metadata.insert(
        "source".to_string(),
        Value::String("ctox-queue".to_string()),
    );
    metadata.insert("priority".to_string(), Value::String(priority.clone()));
    metadata.insert("sort_at".to_string(), Value::String(sort_at.clone()));
    if metadata.get("created_at").is_none() {
        metadata.insert(
            "created_at".to_string(),
            Value::String(current.observed_at.clone()),
        );
    }
    if let Some(skill) = request
        .suggested_skill
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        metadata.insert("skill".to_string(), Value::String(skill.to_string()));
    } else if request.clear_skill {
        metadata.remove("skill");
    }
    if let Some(note) = request
        .status_note
        .as_deref()
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        metadata.insert("status_note".to_string(), Value::String(note.to_string()));
    } else if request.clear_note {
        metadata.remove("status_note");
    }
    upsert_communication_message(
        &mut conn,
        UpsertMessage {
            message_key: &current.message_key,
            channel: QUEUE_CHANNEL_NAME,
            account_key: QUEUE_ACCOUNT_KEY,
            thread_key: &thread_key,
            remote_id: &current.remote_id,
            direction: "inbound",
            folder_hint: "queue",
            sender_display: QUEUE_SENDER_DISPLAY,
            sender_address: QUEUE_SENDER_ADDRESS,
            recipient_addresses_json: "[]",
            cc_addresses_json: "[]",
            bcc_addresses_json: "[]",
            subject: &title,
            preview: &preview_text(&prompt, &title),
            body_text: &prompt,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "high",
            status: "received",
            seen: current.seen,
            has_attachments: false,
            external_created_at: &sort_at,
            observed_at: &now,
            metadata_json: &serde_json::to_string(&metadata)?,
        },
    )?;
    if let Some(route_status) = request.route_status.as_deref() {
        set_routing_status(&mut conn, &current.message_key, route_status, &now)?;
    }
    refresh_thread(&mut conn, &thread_key)?;
    load_queue_task_from_conn(&conn, &current.message_key)?
        .context("failed to load updated queue task")
}

pub fn ingest_cron_message(
    root: &Path,
    run_id: &str,
    thread_key: &str,
    task_name: &str,
    body: &str,
    skill: Option<&str>,
    scheduled_for: &str,
) -> Result<String> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_account(
        &mut conn,
        "cron:system",
        "cron",
        "ctox scheduler",
        "system",
        json!({"source": "cron"}),
    )?;
    let observed_at = now_iso_string();
    let remote_id = format!("cron-{run_id}");
    let message_key = format!("cron:system::{remote_id}");
    let metadata = json!({
        "source": "ctox-schedule",
        "task_name": task_name,
        "skill": skill,
        "scheduled_for": scheduled_for,
        "run_id": run_id,
    });
    upsert_communication_message(
        &mut conn,
        UpsertMessage {
            message_key: &message_key,
            channel: "cron",
            account_key: "cron:system",
            thread_key,
            remote_id: &remote_id,
            direction: "inbound",
            folder_hint: "schedule",
            sender_display: "CTOX scheduler",
            sender_address: "cron:system",
            recipient_addresses_json: "[]",
            cc_addresses_json: "[]",
            bcc_addresses_json: "[]",
            subject: task_name,
            preview: &preview_text(body, task_name),
            body_text: body,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "high",
            status: "received",
            seen: false,
            has_attachments: false,
            external_created_at: scheduled_for,
            observed_at: &observed_at,
            metadata_json: &serde_json::to_string(&metadata)?,
        },
    )?;
    refresh_thread(&mut conn, thread_key)?;
    ensure_routing_rows_for_inbound(&conn)?;
    Ok(message_key)
}

pub fn ingest_plan_message(
    root: &Path,
    goal_id: &str,
    step_id: &str,
    thread_key: &str,
    goal_title: &str,
    step_title: &str,
    body: &str,
    skill: Option<&str>,
    step_order: i64,
    total_steps: i64,
) -> Result<String> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_account(
        &mut conn,
        "plan:system",
        "plan",
        "ctox planner",
        "system",
        json!({"source": "plan"}),
    )?;
    let observed_at = now_iso_string();
    let remote_id = format!("plan-{goal_id}-{step_id}");
    let message_key = format!("plan:system::{goal_id}::{step_id}");
    let metadata = json!({
        "source": "ctox-plan",
        "goal_id": goal_id,
        "step_id": step_id,
        "goal_title": goal_title,
        "step_title": step_title,
        "skill": skill,
        "step_order": step_order,
        "total_steps": total_steps,
    });
    upsert_communication_message(
        &mut conn,
        UpsertMessage {
            message_key: &message_key,
            channel: "plan",
            account_key: "plan:system",
            thread_key,
            remote_id: &remote_id,
            direction: "inbound",
            folder_hint: "plan",
            sender_display: "CTOX planner",
            sender_address: "plan:system",
            recipient_addresses_json: "[]",
            cc_addresses_json: "[]",
            bcc_addresses_json: "[]",
            subject: step_title,
            preview: &preview_text(body, step_title),
            body_text: body,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "high",
            status: "received",
            seen: false,
            has_attachments: false,
            external_created_at: &observed_at,
            observed_at: &observed_at,
            metadata_json: &serde_json::to_string(&metadata)?,
        },
    )?;
    refresh_thread(&mut conn, thread_key)?;
    ensure_routing_rows_for_inbound(&conn)?;
    Ok(message_key)
}

#[derive(Debug, Serialize)]
struct ChannelMessageView {
    message_key: String,
    channel: String,
    account_key: String,
    thread_key: String,
    remote_id: String,
    direction: String,
    folder_hint: String,
    sender_display: String,
    sender_address: String,
    subject: String,
    preview: String,
    body_text: String,
    status: String,
    seen: bool,
    external_created_at: String,
    observed_at: String,
    metadata: Value,
    routing: RoutingView,
}

#[derive(Debug, Clone, Serialize)]
pub struct RoutedInboundMessage {
    pub message_key: String,
    pub channel: String,
    pub account_key: String,
    pub thread_key: String,
    pub sender_display: String,
    pub sender_address: String,
    pub subject: String,
    pub preview: String,
    pub body_text: String,
    pub external_created_at: String,
    pub metadata: Value,
    pub preferred_reply_modality: Option<String>,
}

#[derive(Debug, Serialize)]
struct RoutingView {
    route_status: String,
    lease_owner: Option<String>,
    leased_at: Option<String>,
    acked_at: Option<String>,
    updated_at: String,
}

#[derive(Debug)]
struct TuiIngestRequest {
    account_key: String,
    thread_key: String,
    body: String,
    subject: String,
    sender_display: String,
    sender_address: String,
    metadata: Value,
}

#[derive(Debug)]
struct ChannelSendRequest {
    channel: String,
    account_key: String,
    thread_key: String,
    body: String,
    subject: String,
    to: Vec<String>,
    cc: Vec<String>,
    sender_display: Option<String>,
    sender_address: Option<String>,
    send_voice: bool,
}

fn sync_channel(root: &Path, db_path: &Path, channel: &str, args: &[String]) -> Result<Value> {
    let conn = open_channel_db(db_path)?;
    match channel {
        "email" => {
            let adapter = root.join("scripts/communication_mail_cli.mjs");
            let mut cmd = Command::new(node_binary(root));
            push_mail_runtime_env(root, &mut cmd);
            cmd.arg(&adapter)
                .arg("sync")
                .arg("--db")
                .arg(db_path)
                .arg("--schema")
                .arg(root.join("scripts/communication_schema.sql"));
            forward_adapter_flags(&mut cmd, args, &["--db", "--channel"]);
            let adapter_json = run_json_command(&mut cmd)
                .with_context(|| format!("email sync adapter failed via {}", adapter.display()))?;
            Ok(json!({
                "ok": true,
                "channel": channel,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        "jami" => {
            let adapter = root.join("scripts/communication_jami_cli.mjs");
            let mut cmd = Command::new(node_binary(root));
            cmd.arg(&adapter)
                .arg("sync")
                .arg("--db")
                .arg(db_path)
                .arg("--schema")
                .arg(root.join("scripts/communication_schema.sql"));
            forward_adapter_flags(&mut cmd, args, &["--db", "--channel"]);
            let adapter_json = run_json_command(&mut cmd)
                .with_context(|| format!("jami sync adapter failed via {}", adapter.display()))?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": channel,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        other => anyhow::bail!("unsupported channel sync target: {other}"),
    }
}

fn send_message(root: &Path, db_path: &Path, request: ChannelSendRequest) -> Result<Value> {
    let mut conn = open_channel_db(db_path)?;
    let request = resolve_outbound_subject(&conn, request)?;
    match request.channel.as_str() {
        "tui" => {
            let message_key = store_tui_outbound_message(&mut conn, &request)?;
            Ok(json!({
                "ok": true,
                "channel": "tui",
                "db_path": db_path,
                "message_key": message_key,
                "status": "sent",
            }))
        }
        "email" => {
            let adapter = root.join("scripts/communication_mail_cli.mjs");
            let sender_email = request
                .sender_address
                .clone()
                .unwrap_or_else(|| email_address_from_account_key(&request.account_key));
            let account_config = load_account_config(&conn, &request.account_key)?;
            let mut cmd = Command::new(node_binary(root));
            push_mail_runtime_env(root, &mut cmd);
            cmd.arg(&adapter)
                .arg("send")
                .arg("--db")
                .arg(db_path)
                .arg("--schema")
                .arg(root.join("scripts/communication_schema.sql"))
                .arg("--email")
                .arg(sender_email);
            if let Some(config) = &account_config {
                cmd.arg("--provider").arg(&config.provider);
                push_email_profile_flags(&mut cmd, &config.profile_json);
            }
            cmd.arg("--thread-key").arg(&request.thread_key);
            if !request.to.is_empty() {
                for recipient in &request.to {
                    cmd.arg("--to");
                    cmd.arg(recipient);
                }
            }
            if !request.cc.is_empty() {
                for cc in &request.cc {
                    cmd.arg("--cc");
                    cmd.arg(cc);
                }
            }
            if let Some(display) = &request.sender_display {
                cmd.arg("--profile-name").arg(display);
            }
            cmd.arg("--subject").arg(&request.subject);
            cmd.arg("--body").arg(&request.body);
            let adapter_json = run_json_command(&mut cmd)
                .with_context(|| format!("email send adapter failed via {}", adapter.display()))?;
            Ok(json!({
                "ok": true,
                "channel": "email",
                "db_path": db_path,
                "status": adapter_json
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("accepted"),
                "delivery_confirmed": adapter_json
                    .get("delivery")
                    .and_then(|value| value.get("confirmed"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "adapter_result": adapter_json,
            }))
        }
        "jami" => {
            let adapter = root.join("scripts/communication_jami_cli.mjs");
            let sender = request
                .sender_address
                .clone()
                .unwrap_or_else(|| jami_address_from_account_key(&request.account_key));
            let mut cmd = Command::new(node_binary(root));
            cmd.arg(&adapter)
                .arg("send")
                .arg("--db")
                .arg(db_path)
                .arg("--schema")
                .arg(root.join("scripts/communication_schema.sql"))
                .arg("--account-id")
                .arg(sender)
                .arg("--thread-key")
                .arg(&request.thread_key)
                .arg("--subject")
                .arg(&request.subject)
                .arg("--body")
                .arg(&request.body);
            for recipient in &request.to {
                cmd.arg("--to").arg(recipient);
            }
            if let Some(profile_name) = &request.sender_display {
                cmd.arg("--profile-name").arg(profile_name);
            }
            if request.send_voice {
                cmd.arg("--send-voice");
            }
            let adapter_json = run_json_command(&mut cmd)
                .with_context(|| format!("jami send adapter failed via {}", adapter.display()))?;
            Ok(json!({
                "ok": true,
                "channel": "jami",
                "db_path": db_path,
                "status": adapter_json
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("queued"),
                "delivery_confirmed": adapter_json
                    .get("delivery")
                    .and_then(|value| value.get("confirmed"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "adapter_result": adapter_json,
            }))
        }
        other => anyhow::bail!("unsupported channel send target: {other}"),
    }
}

fn test_channel(
    root: &Path,
    db_path: &Path,
    channel: &str,
    account_key: Option<&str>,
) -> Result<Value> {
    let conn = open_channel_db(db_path)?;
    match channel {
        "tui" => Ok(json!({
            "ok": true,
            "channel": "tui",
            "status": "ready",
            "detail": "local TUI channel does not require external transport setup",
            "db_path": db_path,
        })),
        "email" => {
            let resolved_account_key = resolve_account_key(&conn, "email", account_key)?;
            let account_config =
                load_account_config(&conn, &resolved_account_key)?.ok_or_else(|| {
                    anyhow::anyhow!("missing email account config for {}", resolved_account_key)
                })?;
            let adapter = root.join("scripts/communication_mail_cli.mjs");
            let mut cmd = Command::new(node_binary(root));
            push_mail_runtime_env(root, &mut cmd);
            cmd.arg(&adapter)
                .arg("test")
                .arg("--db")
                .arg(db_path)
                .arg("--schema")
                .arg(root.join("scripts/communication_schema.sql"))
                .arg("--email")
                .arg(email_address_from_account_key(&resolved_account_key))
                .arg("--provider")
                .arg(&account_config.provider);
            push_email_profile_flags(&mut cmd, &account_config.profile_json);
            let adapter_json = run_json_command(&mut cmd)
                .with_context(|| format!("email test adapter failed via {}", adapter.display()))?;
            Ok(json!({
                "ok": adapter_json.get("ok").and_then(Value::as_bool).unwrap_or(false),
                "channel": "email",
                "account_key": resolved_account_key,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        "jami" => {
            let resolved_account_key = resolve_account_key(&conn, "jami", account_key)?;
            let account_config =
                load_account_config(&conn, &resolved_account_key)?.ok_or_else(|| {
                    anyhow::anyhow!("missing jami account config for {}", resolved_account_key)
                })?;
            let adapter = root.join("scripts/communication_jami_cli.mjs");
            let mut cmd = Command::new(node_binary(root));
            cmd.arg(&adapter)
                .arg("test")
                .arg("--db")
                .arg(db_path)
                .arg("--schema")
                .arg(root.join("scripts/communication_schema.sql"))
                .arg("--account-id")
                .arg(jami_address_from_account_key(&resolved_account_key))
                .arg("--provider")
                .arg(&account_config.provider);
            push_jami_profile_flags(&mut cmd, &account_config.profile_json);
            let adapter_json = run_json_command(&mut cmd)
                .with_context(|| format!("jami test adapter failed via {}", adapter.display()))?;
            Ok(json!({
                "ok": adapter_json.get("ok").and_then(Value::as_bool).unwrap_or(false),
                "channel": "jami",
                "account_key": resolved_account_key,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        other => anyhow::bail!("unsupported channel test target: {other}"),
    }
}

fn parse_send_request(args: &[String]) -> Result<ChannelSendRequest> {
    let channel = required_flag_value(args, "--channel")?.to_string();
    let account_key = required_flag_value(args, "--account-key")?.to_string();
    let thread_key = required_flag_value(args, "--thread-key")?.to_string();
    let body = required_flag_value(args, "--body")?.to_string();
    let subject = find_flag_value(args, "--subject")
        .map(ToOwned::to_owned)
        .unwrap_or_default();
    let to = collect_flag_values(args, "--to");
    if channel != "tui" && to.is_empty() {
        anyhow::bail!("channel send for {channel} requires at least one --to value");
    }
    Ok(ChannelSendRequest {
        channel,
        account_key,
        thread_key,
        body,
        subject,
        to,
        cc: collect_flag_values(args, "--cc"),
        sender_display: find_flag_value(args, "--sender-display").map(ToOwned::to_owned),
        sender_address: find_flag_value(args, "--sender-address").map(ToOwned::to_owned),
        send_voice: has_flag(args, "--send-voice"),
    })
}

fn resolve_outbound_subject(
    conn: &Connection,
    mut request: ChannelSendRequest,
) -> Result<ChannelSendRequest> {
    let subject = request.subject.trim();
    if !subject_is_placeholder(subject) {
        return Ok(request);
    }
    if let Some(existing) = load_thread_subject(conn, &request.thread_key)? {
        request.subject = existing;
    }
    if request.channel == "email" && subject_is_placeholder(request.subject.trim()) {
        anyhow::bail!(
            "email send requires a real subject or an existing thread subject for {}",
            request.thread_key
        );
    }
    Ok(request)
}

fn load_thread_subject(conn: &Connection, thread_key: &str) -> Result<Option<String>> {
    Ok(conn
        .query_row(
        "SELECT subject FROM communication_threads WHERE thread_key = ?1 LIMIT 1",
        params![thread_key],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .context("failed to load existing thread subject")?
    .filter(|subject| !subject_is_placeholder(subject.trim())))
}

fn subject_is_placeholder(subject: &str) -> bool {
    let normalized = subject.trim().to_ascii_lowercase();
    normalized.is_empty()
        || normalized == "(no subject)"
        || normalized == "(ohne betreff)"
}

fn parse_tui_ingest_request(args: &[String]) -> Result<TuiIngestRequest> {
    Ok(TuiIngestRequest {
        account_key: required_flag_value(args, "--account-key")?.to_string(),
        thread_key: required_flag_value(args, "--thread-key")?.to_string(),
        body: required_flag_value(args, "--body")?.to_string(),
        subject: find_flag_value(args, "--subject")
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| "TUI input".to_string()),
        sender_display: find_flag_value(args, "--sender-display")
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| "Local TUI".to_string()),
        sender_address: find_flag_value(args, "--sender-address")
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| "tui:local".to_string()),
        metadata: json!({
            "source": "ctox-channel-ingest-tui",
        }),
    })
}

fn open_channel_db(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create db parent {}", parent.display()))?;
    }
    let conn = Connection::open(path)
        .with_context(|| format!("failed to open channel db {}", path.display()))?;
    conn.busy_timeout(std::time::Duration::from_secs(5))
        .context("failed to configure SQLite busy_timeout for channels")?;
    ensure_schema(&conn)?;
    Ok(conn)
}

fn ensure_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        PRAGMA journal_mode=WAL;
        PRAGMA busy_timeout=5000;

        CREATE TABLE IF NOT EXISTS communication_accounts (
            account_key TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            address TEXT NOT NULL,
            provider TEXT NOT NULL,
            profile_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            last_inbound_ok_at TEXT,
            last_outbound_ok_at TEXT
        );

        CREATE TABLE IF NOT EXISTS communication_threads (
            thread_key TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            account_key TEXT NOT NULL,
            subject TEXT NOT NULL,
            participant_keys_json TEXT NOT NULL,
            last_message_key TEXT NOT NULL,
            last_message_at TEXT NOT NULL,
            message_count INTEGER NOT NULL,
            unread_count INTEGER NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS communication_messages (
            message_key TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            account_key TEXT NOT NULL,
            thread_key TEXT NOT NULL,
            remote_id TEXT NOT NULL,
            direction TEXT NOT NULL,
            folder_hint TEXT NOT NULL,
            sender_display TEXT NOT NULL,
            sender_address TEXT NOT NULL,
            recipient_addresses_json TEXT NOT NULL,
            cc_addresses_json TEXT NOT NULL,
            bcc_addresses_json TEXT NOT NULL,
            subject TEXT NOT NULL,
            preview TEXT NOT NULL,
            body_text TEXT NOT NULL,
            body_html TEXT NOT NULL,
            raw_payload_ref TEXT NOT NULL,
            trust_level TEXT NOT NULL,
            status TEXT NOT NULL,
            seen INTEGER NOT NULL,
            has_attachments INTEGER NOT NULL,
            external_created_at TEXT NOT NULL,
            observed_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_communication_messages_account_time
            ON communication_messages(account_key, external_created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_communication_messages_thread
            ON communication_messages(thread_key, external_created_at DESC);

        CREATE INDEX IF NOT EXISTS idx_communication_messages_channel_remote
            ON communication_messages(channel, account_key, remote_id);

        CREATE TABLE IF NOT EXISTS communication_sync_runs (
            run_key TEXT PRIMARY KEY,
            channel TEXT NOT NULL,
            account_key TEXT NOT NULL,
            folder_hint TEXT NOT NULL,
            started_at TEXT NOT NULL,
            finished_at TEXT NOT NULL,
            ok INTEGER NOT NULL,
            fetched_count INTEGER NOT NULL,
            stored_count INTEGER NOT NULL,
            error_text TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS communication_routing_state (
            message_key TEXT PRIMARY KEY,
            route_status TEXT NOT NULL,
            lease_owner TEXT,
            leased_at TEXT,
            acked_at TEXT,
            last_error TEXT,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS owner_profiles (
            owner_key TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#,
    )
    .context("failed to ensure channel schema")?;
    ensure_routing_rows_for_inbound(conn)?;
    Ok(())
}

fn ensure_routing_rows_for_inbound(conn: &Connection) -> Result<()> {
    conn.execute(
        r#"
        INSERT INTO communication_routing_state (
            message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at
        )
        SELECT
            m.message_key,
            CASE
                WHEN m.direction = 'outbound' THEN 'handled'
                WHEN m.trust_level = 'system_probe' THEN 'handled'
                ELSE 'pending'
            END,
            NULL,
            NULL,
            CASE
                WHEN m.direction = 'outbound' OR m.trust_level = 'system_probe' THEN m.observed_at
                ELSE NULL
            END,
            NULL,
            m.observed_at
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE r.message_key IS NULL
        "#,
        [],
    )
    .context("failed to backfill communication routing state")?;
    conn.execute(
        r#"
        UPDATE communication_routing_state
        SET route_status = 'handled',
            lease_owner = NULL,
            leased_at = NULL,
            acked_at = COALESCE(acked_at, (
                SELECT observed_at
                FROM communication_messages m
                WHERE m.message_key = communication_routing_state.message_key
            )),
            last_error = NULL,
            updated_at = COALESCE((
                SELECT observed_at
                FROM communication_messages m
                WHERE m.message_key = communication_routing_state.message_key
            ), updated_at)
        WHERE message_key IN (
            SELECT message_key
            FROM communication_messages
            WHERE direction = 'inbound'
              AND trust_level = 'system_probe'
        )
          AND route_status <> 'handled'
        "#,
        [],
    )
    .context("failed to normalize routing for system probe messages")?;
    Ok(())
}

fn schema_state(conn: &Connection) -> Result<Value> {
    let inbound_count: i64 = conn.query_row(
        "SELECT COUNT(*) FROM communication_messages WHERE direction = 'inbound'",
        [],
        |row| row.get(0),
    )?;
    let thread_count: i64 =
        conn.query_row("SELECT COUNT(*) FROM communication_threads", [], |row| {
            row.get(0)
        })?;
    Ok(json!({
        "inbound_messages": inbound_count,
        "threads": thread_count,
    }))
}

fn ingest_tui_message(conn: &mut Connection, request: TuiIngestRequest) -> Result<Value> {
    ensure_account(
        conn,
        &request.account_key,
        "tui",
        &request.sender_address,
        "local",
        json!({"source": "tui"}),
    )?;
    let observed_at = now_iso_string();
    let remote_id = format!(
        "tui-{}",
        stable_digest(&format!(
            "{}:{}:{}",
            request.thread_key, request.sender_address, request.body
        ))
    );
    let message_key = format!("{}::{remote_id}", request.account_key);
    upsert_communication_message(
        conn,
        UpsertMessage {
            message_key: &message_key,
            channel: "tui",
            account_key: &request.account_key,
            thread_key: &request.thread_key,
            remote_id: &remote_id,
            direction: "inbound",
            folder_hint: "tui",
            sender_display: &request.sender_display,
            sender_address: &request.sender_address,
            recipient_addresses_json: "[]",
            cc_addresses_json: "[]",
            bcc_addresses_json: "[]",
            subject: &request.subject,
            preview: &preview_text(&request.body, &request.subject),
            body_text: &request.body,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "medium",
            status: "received",
            seen: false,
            has_attachments: false,
            external_created_at: &observed_at,
            observed_at: &observed_at,
            metadata_json: &serde_json::to_string(&request.metadata)?,
        },
    )?;
    refresh_thread(conn, &request.thread_key)?;
    ensure_routing_rows_for_inbound(conn)?;
    Ok(json!({
        "message_key": message_key,
        "thread_key": request.thread_key,
        "channel": "tui",
    }))
}

fn store_tui_outbound_message(
    conn: &mut Connection,
    request: &ChannelSendRequest,
) -> Result<String> {
    ensure_account(
        conn,
        &request.account_key,
        "tui",
        request.sender_address.as_deref().unwrap_or("tui:local"),
        "local",
        json!({"source": "tui"}),
    )?;
    let observed_at = now_iso_string();
    let remote_id = format!(
        "tui-out-{}",
        stable_digest(&format!(
            "{}:{}:{}",
            request.thread_key, request.account_key, observed_at
        ))
    );
    let message_key = format!("{}::{remote_id}", request.account_key);
    let sender_display = request
        .sender_display
        .clone()
        .unwrap_or_else(|| "Local TUI".to_string());
    let sender_address = request
        .sender_address
        .clone()
        .unwrap_or_else(|| "tui:local".to_string());
    upsert_communication_message(
        conn,
        UpsertMessage {
            message_key: &message_key,
            channel: "tui",
            account_key: &request.account_key,
            thread_key: &request.thread_key,
            remote_id: &remote_id,
            direction: "outbound",
            folder_hint: "tui",
            sender_display: &sender_display,
            sender_address: &sender_address,
            recipient_addresses_json: &serde_json::to_string(&request.to)?,
            cc_addresses_json: &serde_json::to_string(&request.cc)?,
            bcc_addresses_json: "[]",
            subject: &request.subject,
            preview: &preview_text(&request.body, &request.subject),
            body_text: &request.body,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "high",
            status: "sent",
            seen: true,
            has_attachments: false,
            external_created_at: &observed_at,
            observed_at: &observed_at,
            metadata_json: r#"{"source":"ctox-tui-send"}"#,
        },
    )?;
    refresh_thread(conn, &request.thread_key)?;
    ensure_routing_rows_for_inbound(conn)?;
    Ok(message_key)
}

fn take_messages(
    conn: &mut Connection,
    channel: Option<&str>,
    limit: usize,
    lease_owner: &str,
) -> Result<Vec<ChannelMessageView>> {
    let sql = if channel.is_some() {
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.account_key,
            m.thread_key,
            m.remote_id,
            m.direction,
            m.folder_hint,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.body_text,
            m.status,
            m.seen,
            m.external_created_at,
            m.observed_at,
            m.metadata_json,
            r.route_status,
            r.lease_owner,
            r.leased_at,
            r.acked_at,
            r.updated_at
        FROM communication_messages m
        JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.direction = 'inbound'
          AND m.channel = ?1
          AND r.route_status IN ('pending', 'leased')
          AND (r.lease_owner IS NULL OR r.lease_owner = '' OR r.lease_owner = ?2)
        ORDER BY m.external_created_at ASC, m.observed_at ASC
        LIMIT ?3
        "#
    } else {
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.account_key,
            m.thread_key,
            m.remote_id,
            m.direction,
            m.folder_hint,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.body_text,
            m.status,
            m.seen,
            m.external_created_at,
            m.observed_at,
            m.metadata_json,
            r.route_status,
            r.lease_owner,
            r.leased_at,
            r.acked_at,
            r.updated_at
        FROM communication_messages m
        JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.direction = 'inbound'
          AND r.route_status IN ('pending', 'leased')
          AND (r.lease_owner IS NULL OR r.lease_owner = '' OR r.lease_owner = ?1)
        ORDER BY m.external_created_at ASC, m.observed_at ASC
        LIMIT ?2
        "#
    };

    let mut statement = conn.prepare(sql)?;
    let mapped = if let Some(channel) = channel {
        statement.query_map(
            params![channel, lease_owner, limit as i64],
            map_channel_message_row,
        )?
    } else {
        statement.query_map(params![lease_owner, limit as i64], map_channel_message_row)?
    };
    let rows = mapped.collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    let tx = conn.unchecked_transaction()?;
    let leased_at = now_iso_string();
    let mut taken = Vec::new();
    for mut item in rows {
        tx.execute(
            r#"
            INSERT INTO communication_routing_state (
                message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at
            )
            VALUES (?1, 'leased', ?2, ?3, NULL, NULL, ?3)
            ON CONFLICT(message_key) DO UPDATE SET
                route_status='leased',
                lease_owner=excluded.lease_owner,
                leased_at=excluded.leased_at,
                updated_at=excluded.updated_at
            "#,
            params![item.message_key, lease_owner, leased_at],
        )?;
        item.routing.route_status = "leased".to_string();
        item.routing.lease_owner = Some(lease_owner.to_string());
        item.routing.leased_at = Some(leased_at.clone());
        item.routing.updated_at = leased_at.clone();
        taken.push(item);
    }
    tx.commit()?;
    Ok(taken)
}

fn ack_messages(conn: &mut Connection, message_keys: &[String], status: &str) -> Result<usize> {
    let now = now_iso_string();
    let tx = conn.unchecked_transaction()?;
    let mut updated = 0usize;
    for message_key in message_keys {
        updated += tx.execute(
            r#"
            INSERT INTO communication_routing_state (
                message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at
            )
            VALUES (?1, ?2, NULL, NULL, ?3, NULL, ?3)
            ON CONFLICT(message_key) DO UPDATE SET
                route_status=excluded.route_status,
                lease_owner=NULL,
                leased_at=NULL,
                acked_at=excluded.acked_at,
                updated_at=excluded.updated_at
            "#,
            params![message_key, status, now],
        )?;
        tx.execute(
            "UPDATE communication_messages SET seen = 1 WHERE message_key = ?1",
            params![message_key],
        )?;
    }
    tx.commit()?;
    Ok(updated)
}

fn list_messages(
    conn: &Connection,
    channel: Option<&str>,
    limit: usize,
) -> Result<Vec<ChannelMessageView>> {
    let sql = if channel.is_some() {
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.account_key,
            m.thread_key,
            m.remote_id,
            m.direction,
            m.folder_hint,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.body_text,
            m.status,
            m.seen,
            m.external_created_at,
            m.observed_at,
            m.metadata_json,
            COALESCE(r.route_status, 'pending'),
            r.lease_owner,
            r.leased_at,
            r.acked_at,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = ?1
        ORDER BY m.external_created_at DESC, m.observed_at DESC
        LIMIT ?2
        "#
    } else {
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.account_key,
            m.thread_key,
            m.remote_id,
            m.direction,
            m.folder_hint,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.body_text,
            m.status,
            m.seen,
            m.external_created_at,
            m.observed_at,
            m.metadata_json,
            COALESCE(r.route_status, 'pending'),
            r.lease_owner,
            r.leased_at,
            r.acked_at,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        ORDER BY m.external_created_at DESC, m.observed_at DESC
        LIMIT ?1
        "#
    };
    let mut statement = conn.prepare(sql)?;
    let rows = if let Some(channel) = channel {
        statement.query_map(params![channel, limit as i64], map_channel_message_row)?
    } else {
        statement.query_map(params![limit as i64], map_channel_message_row)?
    };
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

fn list_thread_messages(
    conn: &Connection,
    thread_key: &str,
    limit: usize,
) -> Result<Vec<ChannelMessageView>> {
    let mut statement = conn.prepare(
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.account_key,
            m.thread_key,
            m.remote_id,
            m.direction,
            m.folder_hint,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.body_text,
            m.status,
            m.seen,
            m.external_created_at,
            m.observed_at,
            m.metadata_json,
            COALESCE(r.route_status, 'pending'),
            r.lease_owner,
            r.leased_at,
            r.acked_at,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.thread_key = ?1
        ORDER BY m.external_created_at DESC, m.observed_at DESC
        LIMIT ?2
        "#,
    )?;
    let rows = statement.query_map(params![thread_key, limit as i64], map_channel_message_row)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

fn search_messages(
    conn: &Connection,
    query: &str,
    channel: Option<&str>,
    sender: Option<&str>,
    limit: usize,
) -> Result<Vec<ChannelMessageView>> {
    let normalized_query = format!("%{}%", query.trim().to_ascii_lowercase());
    let normalized_sender = sender
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| !value.is_empty());
    let mut statement = conn.prepare(
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.account_key,
            m.thread_key,
            m.remote_id,
            m.direction,
            m.folder_hint,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.body_text,
            m.status,
            m.seen,
            m.external_created_at,
            m.observed_at,
            m.metadata_json,
            COALESCE(r.route_status, 'pending'),
            r.lease_owner,
            r.leased_at,
            r.acked_at,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE
            (?1 IS NULL OR m.channel = ?1)
            AND (?2 IS NULL OR LOWER(m.sender_address) = ?2)
            AND (
                LOWER(m.subject) LIKE ?3
                OR LOWER(m.preview) LIKE ?3
                OR LOWER(m.body_text) LIKE ?3
                OR LOWER(m.sender_display) LIKE ?3
                OR LOWER(m.sender_address) LIKE ?3
                OR LOWER(m.thread_key) LIKE ?3
            )
        ORDER BY m.external_created_at DESC, m.observed_at DESC
        LIMIT ?4
        "#,
    )?;
    let rows = statement.query_map(
        params![channel, normalized_sender, normalized_query, limit as i64],
        map_channel_message_row,
    )?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

fn map_channel_message_row(row: &rusqlite::Row<'_>) -> rusqlite::Result<ChannelMessageView> {
    let metadata_json: String = row.get(16)?;
    let metadata = serde_json::from_str(&metadata_json)
        .unwrap_or_else(|_| json!({"raw_metadata": metadata_json}));
    Ok(ChannelMessageView {
        message_key: row.get(0)?,
        channel: row.get(1)?,
        account_key: row.get(2)?,
        thread_key: row.get(3)?,
        remote_id: row.get(4)?,
        direction: row.get(5)?,
        folder_hint: row.get(6)?,
        sender_display: row.get(7)?,
        sender_address: row.get(8)?,
        subject: row.get(9)?,
        preview: row.get(10)?,
        body_text: row.get(11)?,
        status: row.get(12)?,
        seen: row.get::<_, i64>(13)? != 0,
        external_created_at: row.get(14)?,
        observed_at: row.get(15)?,
        metadata,
        routing: RoutingView {
            route_status: row.get(17)?,
            lease_owner: row.get(18)?,
            leased_at: row.get(19)?,
            acked_at: row.get(20)?,
            updated_at: row.get(21)?,
        },
    })
}

fn list_queue_tasks_from_conn(conn: &Connection, limit: usize) -> Result<Vec<QueueTaskView>> {
    let mut statement = conn.prepare(
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.account_key,
            m.thread_key,
            m.remote_id,
            m.direction,
            m.folder_hint,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.body_text,
            m.status,
            m.seen,
            m.external_created_at,
            m.observed_at,
            m.metadata_json,
            COALESCE(r.route_status, 'pending'),
            r.lease_owner,
            r.leased_at,
            r.acked_at,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = ?1
          AND m.direction = 'inbound'
        ORDER BY
            CASE COALESCE(r.route_status, 'pending')
                WHEN 'pending' THEN 0
                WHEN 'leased' THEN 1
                WHEN 'blocked' THEN 2
                WHEN 'failed' THEN 3
                WHEN 'handled' THEN 4
                WHEN 'cancelled' THEN 5
                ELSE 9
            END ASC,
            m.external_created_at ASC,
            m.observed_at ASC
        LIMIT ?2
        "#,
    )?;
    let rows = statement.query_map(
        params![QUEUE_CHANNEL_NAME, limit as i64],
        map_channel_message_row,
    )?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)?
        .into_iter()
        .map(queue_task_from_message)
        .collect()
}

fn load_queue_message_from_conn(
    conn: &Connection,
    message_key: &str,
) -> Result<Option<ChannelMessageView>> {
    conn.query_row(
        r#"
        SELECT
            m.message_key,
            m.channel,
            m.account_key,
            m.thread_key,
            m.remote_id,
            m.direction,
            m.folder_hint,
            m.sender_display,
            m.sender_address,
            m.subject,
            m.preview,
            m.body_text,
            m.status,
            m.seen,
            m.external_created_at,
            m.observed_at,
            m.metadata_json,
            COALESCE(r.route_status, 'pending'),
            r.lease_owner,
            r.leased_at,
            r.acked_at,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = ?1
          AND m.direction = 'inbound'
          AND m.message_key = ?2
        LIMIT 1
        "#,
        params![QUEUE_CHANNEL_NAME, message_key],
        map_channel_message_row,
    )
    .optional()
    .map_err(anyhow::Error::from)
}

fn load_queue_task_from_conn(
    conn: &Connection,
    message_key: &str,
) -> Result<Option<QueueTaskView>> {
    load_queue_message_from_conn(conn, message_key)?
        .map(queue_task_from_message)
        .transpose()
}

fn queue_task_from_message(message: ChannelMessageView) -> Result<QueueTaskView> {
    if message.channel != QUEUE_CHANNEL_NAME || message.direction != "inbound" {
        anyhow::bail!("message is not a queue task");
    }
    let priority = current_queue_priority(&message);
    let created_at = message
        .metadata
        .get("created_at")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| message.observed_at.clone());
    let sort_at = message
        .metadata
        .get("sort_at")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| message.external_created_at.clone());
    Ok(QueueTaskView {
        message_key: message.message_key,
        thread_key: message.thread_key,
        title: message.subject,
        prompt: message.body_text,
        priority,
        suggested_skill: message
            .metadata
            .get("skill")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        parent_message_key: message
            .metadata
            .get("parent_message_key")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        route_status: message.routing.route_status,
        status_note: message
            .metadata
            .get("status_note")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
        lease_owner: message.routing.lease_owner,
        leased_at: message.routing.leased_at,
        acked_at: message.routing.acked_at,
        created_at,
        sort_at,
        updated_at: message.routing.updated_at,
    })
}

fn ensure_queue_account(conn: &mut Connection) -> Result<()> {
    ensure_account(
        conn,
        QUEUE_ACCOUNT_KEY,
        QUEUE_CHANNEL_NAME,
        QUEUE_ACCOUNT_ADDRESS,
        QUEUE_PROVIDER,
        json!({"source": "ctox-queue"}),
    )
}

fn set_routing_status(
    conn: &mut Connection,
    message_key: &str,
    route_status: &str,
    now: &str,
) -> Result<()> {
    let route_status = canonical_queue_route_status(route_status)?;
    let acked_at = if matches!(route_status.as_str(), "handled" | "cancelled") {
        Some(now)
    } else {
        None
    };
    conn.execute(
        r#"
        INSERT INTO communication_routing_state (
            message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at
        )
        VALUES (?1, ?2, NULL, NULL, ?3, NULL, ?4)
        ON CONFLICT(message_key) DO UPDATE SET
            route_status=excluded.route_status,
            lease_owner=NULL,
            leased_at=NULL,
            acked_at=excluded.acked_at,
            updated_at=excluded.updated_at
        "#,
        params![message_key, route_status, acked_at, now],
    )?;
    Ok(())
}

fn canonical_queue_priority(raw: &str) -> Result<String> {
    let normalized = raw.trim().to_lowercase();
    match normalized.as_str() {
        "urgent" | "high" | "normal" | "low" => Ok(normalized),
        _ => anyhow::bail!("unsupported queue priority '{raw}' (expected urgent|high|normal|low)"),
    }
}

fn canonical_queue_route_status(raw: &str) -> Result<String> {
    let normalized = raw.trim().to_lowercase();
    match normalized.as_str() {
        "pending" | "blocked" | "failed" | "handled" | "cancelled" => Ok(normalized),
        _ => anyhow::bail!(
            "unsupported queue route status '{raw}' (expected pending|blocked|failed|handled|cancelled)"
        ),
    }
}

fn current_queue_priority(message: &ChannelMessageView) -> String {
    message
        .metadata
        .get("priority")
        .and_then(Value::as_str)
        .unwrap_or("normal")
        .trim()
        .to_lowercase()
}

fn queue_metadata_object(metadata: &Value) -> serde_json::Map<String, Value> {
    metadata
        .as_object()
        .cloned()
        .unwrap_or_else(serde_json::Map::new)
}

fn queue_sort_at(priority: &str, now: &str) -> Result<String> {
    let base = DateTime::parse_from_rfc3339(now)
        .with_context(|| format!("failed to parse queue timestamp '{now}'"))?
        .with_timezone(&Utc);
    let shifted = match priority {
        "urgent" => base - Duration::hours(24),
        "high" => base - Duration::hours(1),
        "normal" => base,
        "low" => base + Duration::hours(1),
        _ => anyhow::bail!("unsupported queue priority '{priority}'"),
    };
    Ok(shifted.to_rfc3339())
}

struct UpsertMessage<'a> {
    message_key: &'a str,
    channel: &'a str,
    account_key: &'a str,
    thread_key: &'a str,
    remote_id: &'a str,
    direction: &'a str,
    folder_hint: &'a str,
    sender_display: &'a str,
    sender_address: &'a str,
    recipient_addresses_json: &'a str,
    cc_addresses_json: &'a str,
    bcc_addresses_json: &'a str,
    subject: &'a str,
    preview: &'a str,
    body_text: &'a str,
    body_html: &'a str,
    raw_payload_ref: &'a str,
    trust_level: &'a str,
    status: &'a str,
    seen: bool,
    has_attachments: bool,
    external_created_at: &'a str,
    observed_at: &'a str,
    metadata_json: &'a str,
}

fn upsert_communication_message(conn: &mut Connection, message: UpsertMessage<'_>) -> Result<()> {
    let tx = conn.unchecked_transaction()?;
    upsert_communication_message_tx(&tx, message)?;
    tx.commit()?;
    Ok(())
}

fn upsert_communication_message_tx(tx: &Transaction<'_>, message: UpsertMessage<'_>) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO communication_messages (
            message_key, channel, account_key, thread_key, remote_id, direction, folder_hint,
            sender_display, sender_address, recipient_addresses_json, cc_addresses_json, bcc_addresses_json,
            subject, preview, body_text, body_html, raw_payload_ref, trust_level, status, seen,
            has_attachments, external_created_at, observed_at, metadata_json
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6, ?7,
            ?8, ?9, ?10, ?11, ?12,
            ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20,
            ?21, ?22, ?23, ?24
        )
        ON CONFLICT(message_key) DO UPDATE SET
            channel=excluded.channel,
            account_key=excluded.account_key,
            thread_key=excluded.thread_key,
            remote_id=excluded.remote_id,
            direction=excluded.direction,
            folder_hint=excluded.folder_hint,
            sender_display=excluded.sender_display,
            sender_address=excluded.sender_address,
            recipient_addresses_json=excluded.recipient_addresses_json,
            cc_addresses_json=excluded.cc_addresses_json,
            bcc_addresses_json=excluded.bcc_addresses_json,
            subject=excluded.subject,
            preview=excluded.preview,
            body_text=excluded.body_text,
            body_html=excluded.body_html,
            raw_payload_ref=excluded.raw_payload_ref,
            trust_level=excluded.trust_level,
            status=excluded.status,
            seen=excluded.seen,
            has_attachments=excluded.has_attachments,
            external_created_at=excluded.external_created_at,
            observed_at=excluded.observed_at,
            metadata_json=excluded.metadata_json
        "#,
        params![
            message.message_key,
            message.channel,
            message.account_key,
            message.thread_key,
            message.remote_id,
            message.direction,
            message.folder_hint,
            message.sender_display,
            message.sender_address,
            message.recipient_addresses_json,
            message.cc_addresses_json,
            message.bcc_addresses_json,
            message.subject,
            message.preview,
            message.body_text,
            message.body_html,
            message.raw_payload_ref,
            message.trust_level,
            message.status,
            if message.seen { 1 } else { 0 },
            if message.has_attachments { 1 } else { 0 },
            message.external_created_at,
            message.observed_at,
            message.metadata_json,
        ],
    )?;
    Ok(())
}

fn ensure_account(
    conn: &mut Connection,
    account_key: &str,
    channel: &str,
    address: &str,
    provider: &str,
    profile_json: Value,
) -> Result<()> {
    let tx = conn.unchecked_transaction()?;
    ensure_account_tx(&tx, account_key, channel, address, provider, profile_json)?;
    tx.commit()?;
    Ok(())
}

fn upsert_owner_profile(conn: &mut Connection, display_name: &str) -> Result<()> {
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO owner_profiles (
            owner_key, display_name, metadata_json, created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?4)
        ON CONFLICT(owner_key) DO UPDATE SET
            display_name=excluded.display_name,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        "#,
        params!["primary", display_name.trim(), r#"{}"#, now],
    )?;
    Ok(())
}

fn sync_identity_profiles(conn: &mut Connection, settings: &BTreeMap<String, String>) -> Result<()> {
    let owner_name = settings
        .get("CTOX_OWNER_NAME")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("Owner");
    if let Some(owner_email) = settings
        .get("CTOX_OWNER_EMAIL_ADDRESS")
        .map(|value| normalize_email_address(value))
        .filter(|value| !value.is_empty())
    {
        upsert_identity_profile(
            conn,
            &owner_email,
            owner_name,
            json!({
                "email": owner_email,
                "role": "owner",
                "allow_admin_actions": true,
                "allow_sudo_actions": true,
                "mail_instruction_scope": "full_admin",
            }),
        )?;
    }

    for admin in parse_admin_email_policies(settings) {
        upsert_identity_profile(
            conn,
            &admin.email,
            &admin.email,
            json!({
                "email": admin.email,
                "role": "admin",
                "allow_admin_actions": true,
                "allow_sudo_actions": admin.can_sudo,
                "mail_instruction_scope": if admin.can_sudo { "admin_with_sudo" } else { "admin_without_sudo" },
            }),
        )?;
    }
    Ok(())
}

fn upsert_identity_profile(
    conn: &mut Connection,
    owner_key: &str,
    display_name: &str,
    metadata: Value,
) -> Result<()> {
    let now = now_iso_string();
    conn.execute(
        r#"
        INSERT INTO owner_profiles (
            owner_key, display_name, metadata_json, created_at, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?4)
        ON CONFLICT(owner_key) DO UPDATE SET
            display_name=excluded.display_name,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        "#,
        params![
            owner_key,
            display_name.trim(),
            serde_json::to_string(&metadata)?,
            now
        ],
    )?;
    Ok(())
}

fn load_owner_name(conn: &Connection) -> Result<Option<String>> {
    Ok(conn
        .query_row(
            r#"
        SELECT display_name
        FROM owner_profiles
        WHERE owner_key = 'primary'
        LIMIT 1
        "#,
            [],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .filter(|name| !name.trim().is_empty()))
}

fn admin_email_policy_summaries(settings: &BTreeMap<String, String>) -> Vec<String> {
    let admins = parse_admin_email_policies(settings);
    if admins.is_empty() {
        return vec!["- no additional admin mail profiles configured".to_string()];
    }
    admins
        .into_iter()
        .map(|entry| {
            format!(
                "- {} ({})",
                entry.email,
                if entry.can_sudo {
                    "admin with sudo"
                } else {
                    "admin without sudo"
                }
            )
        })
        .collect()
}

fn parse_admin_email_policies(settings: &BTreeMap<String, String>) -> Vec<AdminEmailPolicy> {
    let raw = settings
        .get("CTOX_EMAIL_ADMIN_POLICIES")
        .map(String::as_str)
        .unwrap_or("");
    let mut seen = BTreeSet::new();
    let mut out = Vec::new();
    for entry in raw
        .split(|ch| matches!(ch, '\n' | ',' | ';'))
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let separator_index = entry.find(['|', ':', '=']);
        let (email_part, policy_part) = if let Some(index) = separator_index {
            (entry[..index].trim(), entry[index + 1..].trim())
        } else {
            (entry, "")
        };
        let email = normalize_email_address(email_part);
        if email.is_empty() || !seen.insert(email.clone()) {
            continue;
        }
        let policy = policy_part.to_ascii_lowercase().replace(' ', "");
        let can_sudo = policy == "sudo"
            || policy == "admin+sudo"
            || policy == "withsudo"
            || (policy.contains("sudo")
                && !policy.contains("no-sudo")
                && !policy.contains("nosudo")
                && !policy.contains("withoutsudo"));
        out.push(AdminEmailPolicy { email, can_sudo });
    }
    out
}

fn normalize_email_address(value: &str) -> String {
    value.trim().trim_matches('<').trim_matches('>').to_lowercase()
}

fn normalized_allowed_email_domain(settings: &BTreeMap<String, String>) -> Option<String> {
    settings
        .get("CTOX_ALLOWED_EMAIL_DOMAIN")
        .map(|value| value.trim().trim_start_matches('@').to_lowercase())
        .filter(|value| !value.is_empty())
        .or_else(|| {
            settings
                .get("CTOX_OWNER_EMAIL_ADDRESS")
                .map(|value| normalize_email_address(value))
                .and_then(|value| value.split_once('@').map(|(_, domain)| domain.to_string()))
                .filter(|value| !value.is_empty())
        })
}

fn email_matches_domain(email: &str, domain: &str) -> bool {
    email
        .rsplit_once('@')
        .map(|(_, candidate_domain)| candidate_domain.eq_ignore_ascii_case(domain))
        .unwrap_or(false)
}

fn ensure_account_tx(
    tx: &Transaction<'_>,
    account_key: &str,
    channel: &str,
    address: &str,
    provider: &str,
    profile_json: Value,
) -> Result<()> {
    let now = now_iso_string();
    tx.execute(
        r#"
        INSERT INTO communication_accounts (
            account_key, channel, address, provider, profile_json, created_at, updated_at, last_inbound_ok_at, last_outbound_ok_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?6, NULL, NULL)
        ON CONFLICT(account_key) DO UPDATE SET
            channel=excluded.channel,
            address=excluded.address,
            provider=excluded.provider,
            profile_json=excluded.profile_json,
            updated_at=excluded.updated_at
        "#,
        params![
            account_key,
            channel,
            address,
            provider,
            serde_json::to_string(&profile_json)?,
            now,
        ],
    )?;
    Ok(())
}

fn refresh_thread(conn: &mut Connection, thread_key: &str) -> Result<()> {
    let tx = conn.unchecked_transaction()?;
    refresh_thread_tx(&tx, thread_key)?;
    tx.commit()?;
    Ok(())
}

fn refresh_thread_tx(tx: &Transaction<'_>, thread_key: &str) -> Result<()> {
    let summary = tx
        .query_row(
            r#"
            SELECT
                channel,
                account_key,
                subject,
                message_key,
                external_created_at
            FROM communication_messages
            WHERE thread_key = ?1
            ORDER BY external_created_at DESC, observed_at DESC
            LIMIT 1
            "#,
            params![thread_key],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, String>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()?;
    let Some((channel, account_key, subject, last_message_key, last_message_at)) = summary else {
        return Ok(());
    };

    let message_count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM communication_messages WHERE thread_key = ?1",
        params![thread_key],
        |row| row.get(0),
    )?;
    let unread_count: i64 = tx.query_row(
        "SELECT COUNT(*) FROM communication_messages WHERE thread_key = ?1 AND direction = 'inbound' AND seen = 0",
        params![thread_key],
        |row| row.get(0),
    )?;
    let mut participants = BTreeSet::new();
    let mut participant_stmt = tx.prepare(
        r#"
        SELECT sender_address, recipient_addresses_json, cc_addresses_json
        FROM communication_messages
        WHERE thread_key = ?1
        "#,
    )?;
    let participant_rows = participant_stmt.query_map(params![thread_key], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    for row in participant_rows {
        let (sender, recipients_json, cc_json) = row?;
        if !sender.trim().is_empty() {
            participants.insert(sender);
        }
        for value in parse_string_json_array(&recipients_json) {
            participants.insert(value);
        }
        for value in parse_string_json_array(&cc_json) {
            participants.insert(value);
        }
    }

    tx.execute(
        r#"
        INSERT INTO communication_threads (
            thread_key, channel, account_key, subject, participant_keys_json, last_message_key,
            last_message_at, message_count, unread_count, metadata_json, updated_at
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
        ON CONFLICT(thread_key) DO UPDATE SET
            channel=excluded.channel,
            account_key=excluded.account_key,
            subject=excluded.subject,
            participant_keys_json=excluded.participant_keys_json,
            last_message_key=excluded.last_message_key,
            last_message_at=excluded.last_message_at,
            message_count=excluded.message_count,
            unread_count=excluded.unread_count,
            metadata_json=excluded.metadata_json,
            updated_at=excluded.updated_at
        "#,
        params![
            thread_key,
            channel,
            account_key,
            subject,
            serde_json::to_string(&participants.into_iter().collect::<Vec<_>>())?,
            last_message_key,
            last_message_at,
            message_count,
            unread_count,
            r#"{"refreshed_by":"ctox-channel-router"}"#,
            now_iso_string(),
        ],
    )?;
    Ok(())
}

fn ensure_routing_rows_for_inbound_tx(tx: &Transaction<'_>) -> Result<()> {
    tx.execute(
        r#"
        INSERT INTO communication_routing_state (
            message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at
        )
        SELECT
            m.message_key,
            CASE WHEN m.direction = 'outbound' THEN 'handled' ELSE 'pending' END,
            NULL,
            NULL,
            CASE WHEN m.direction = 'outbound' THEN m.observed_at ELSE NULL END,
            NULL,
            m.observed_at
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE r.message_key IS NULL
        "#,
        [],
    )?;
    Ok(())
}

fn preview_text(body: &str, subject: &str) -> String {
    let source = if body.trim().is_empty() {
        subject
    } else {
        body
    };
    let collapsed = source.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.chars().take(280).collect()
}

fn parse_string_json_array(raw: &str) -> Vec<String> {
    serde_json::from_str::<Vec<String>>(raw).unwrap_or_default()
}

fn stable_digest(input: &str) -> String {
    let digest = Sha256::digest(input.as_bytes());
    let hex = format!("{digest:x}");
    hex[..24].to_string()
}

fn email_address_from_account_key(account_key: &str) -> String {
    account_key
        .strip_prefix("email:")
        .unwrap_or(account_key)
        .to_string()
}

fn node_binary(root: &Path) -> String {
    runtime_config::load_runtime_env_map(root)
        .ok()
        .and_then(|settings| settings.get("CTOX_NODE_BIN").cloned())
        .map(|value| value.trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "node".to_string())
}

fn push_mail_runtime_env(root: &Path, cmd: &mut Command) {
    let Ok(settings) = runtime_config::load_runtime_env_map(root) else {
        return;
    };
    for key in [
        "CTO_EMAIL_PASSWORD",
        "CTO_EMAIL_GRAPH_ACCESS_TOKEN",
        "CTO_EMAIL_GRAPH_BASE_URL",
        "CTO_EMAIL_GRAPH_USER",
        "CTO_EMAIL_EWS_URL",
        "CTO_EMAIL_OWA_URL",
        "CTO_EMAIL_EWS_VERSION",
        "CTO_EMAIL_EWS_AUTH_TYPE",
        "CTO_EMAIL_EWS_USERNAME",
        "CTO_EMAIL_EWS_BEARER_TOKEN",
        "CTO_EMAIL_ACTIVESYNC_SERVER",
        "CTO_EMAIL_ACTIVESYNC_USERNAME",
        "CTO_EMAIL_ACTIVESYNC_PATH",
        "CTO_EMAIL_ACTIVESYNC_DEVICE_ID",
        "CTO_EMAIL_ACTIVESYNC_DEVICE_TYPE",
        "CTO_EMAIL_ACTIVESYNC_PROTOCOL_VERSION",
        "CTO_EMAIL_ACTIVESYNC_POLICY_KEY",
        "CTO_EMAIL_VERIFY_SEND",
        "CTO_EMAIL_SENT_VERIFY_WINDOW_SECONDS",
    ] {
        if let Some(value) = settings
            .get(key)
            .map(|value| value.trim())
            .filter(|value| !value.is_empty())
        {
            cmd.env(key, value);
        }
    }
}

#[derive(Debug)]
struct AccountConfig {
    provider: String,
    profile_json: Value,
}

fn load_account_config(conn: &Connection, account_key: &str) -> Result<Option<AccountConfig>> {
    let row = conn
        .query_row(
            r#"
            SELECT provider, profile_json
            FROM communication_accounts
            WHERE account_key = ?1
            LIMIT 1
            "#,
            params![account_key],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?;
    let Some((provider, profile_json)) = row else {
        return Ok(None);
    };
    let parsed_profile = serde_json::from_str(&profile_json)
        .unwrap_or_else(|_| json!({ "raw_profile_json": profile_json }));
    Ok(Some(AccountConfig {
        provider,
        profile_json: parsed_profile,
    }))
}

fn push_email_profile_flags(cmd: &mut Command, profile_json: &Value) {
    let Some(object) = profile_json.as_object() else {
        return;
    };
    let mappings = [
        ("imapHost", "--imap-host"),
        ("imapPort", "--imap-port"),
        ("smtpHost", "--smtp-host"),
        ("smtpPort", "--smtp-port"),
        ("folder", "--folder"),
        ("graphBaseUrl", "--graph-base-url"),
        ("graphUser", "--graph-user"),
        ("ewsUrl", "--ews-url"),
        ("owaUrl", "--owa-url"),
        ("ewsVersion", "--ews-version"),
        ("ewsAuthType", "--ews-auth-type"),
        ("ewsUsername", "--ews-username"),
        ("activeSyncServer", "--active-sync-server"),
        ("activeSyncUsername", "--active-sync-username"),
        ("activeSyncPath", "--active-sync-path"),
        ("activeSyncDeviceId", "--active-sync-device-id"),
        ("activeSyncDeviceType", "--active-sync-device-type"),
        (
            "activeSyncProtocolVersion",
            "--active-sync-protocol-version",
        ),
        ("activeSyncPolicyKey", "--active-sync-policy-key"),
    ];
    for (json_key, flag) in mappings {
        let Some(value) = object.get(json_key) else {
            continue;
        };
        if value.is_null() {
            continue;
        }
        let rendered = match value {
            Value::String(text) => text.trim().to_string(),
            Value::Number(number) => number.to_string(),
            Value::Bool(boolean) => {
                if *boolean {
                    "true".to_string()
                } else {
                    "false".to_string()
                }
            }
            _ => continue,
        };
        if rendered.is_empty() {
            continue;
        }
        cmd.arg(flag).arg(rendered);
    }
}

fn push_jami_profile_flags(cmd: &mut Command, profile_json: &Value) {
    let Some(object) = profile_json.as_object() else {
        return;
    };
    let mappings = [
        ("accountId", "--account-id"),
        ("profileName", "--profile-name"),
        ("inboxDir", "--inbox-dir"),
        ("outboxDir", "--outbox-dir"),
        ("archiveDir", "--archive-dir"),
        ("dbusEnvFile", "--dbus-env-file"),
    ];
    for (field, flag) in mappings {
        let Some(value) = object.get(field).and_then(Value::as_str) else {
            continue;
        };
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        cmd.arg(flag).arg(trimmed);
    }
}

fn jami_address_from_account_key(account_key: &str) -> String {
    account_key
        .strip_prefix("jami:")
        .unwrap_or(account_key)
        .to_string()
}

fn resolve_account_key(conn: &Connection, channel: &str, explicit: Option<&str>) -> Result<String> {
    if let Some(value) = explicit.map(str::trim).filter(|value| !value.is_empty()) {
        return Ok(value.to_string());
    }
    conn.query_row(
        r#"
        SELECT account_key
        FROM communication_accounts
        WHERE channel = ?1
        ORDER BY updated_at DESC, account_key ASC
        LIMIT 1
        "#,
        params![channel],
        |row| row.get::<_, String>(0),
    )
    .optional()?
    .ok_or_else(|| anyhow::anyhow!("no configured account found for channel {channel}"))
}

fn resolve_db_path(root: &Path, explicit: Option<&str>) -> PathBuf {
    explicit
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join(DEFAULT_DB_RELATIVE_PATH))
}

fn required_flag_value<'a>(args: &'a [String], flag: &str) -> Result<&'a str> {
    find_flag_value(args, flag).with_context(|| format!("missing required flag {flag}"))
}

fn find_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == flag {
            return args.get(index + 1).map(String::as_str);
        }
        index += 1;
    }
    None
}

fn collect_flag_values(args: &[String], flag: &str) -> Vec<String> {
    let mut values = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        if args[index] == flag {
            if let Some(value) = args.get(index + 1) {
                values.push(value.clone());
            }
            index += 2;
        } else {
            index += 1;
        }
    }
    values
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|arg| arg == flag)
}

fn positional_after_flags(args: &[String]) -> Vec<String> {
    let mut items = Vec::new();
    let mut index = 0usize;
    while index < args.len() {
        let token = &args[index];
        if token.starts_with("--") {
            index += 1;
            if index < args.len() && !args[index].starts_with("--") {
                index += 1;
            }
            continue;
        }
        items.push(token.clone());
        index += 1;
    }
    items
}

fn forward_adapter_flags(cmd: &mut Command, args: &[String], skip_flags: &[&str]) {
    let mut index = 0usize;
    while index < args.len() {
        let token = &args[index];
        if skip_flags.iter().any(|flag| token == flag) {
            index += 2;
            continue;
        }
        if token.starts_with("--") {
            cmd.arg(token);
            if let Some(value) = args.get(index + 1) {
                if !value.starts_with("--") {
                    cmd.arg(value);
                    index += 2;
                    continue;
                }
            }
        }
        index += 1;
    }
}

fn run_json_command(cmd: &mut Command) -> Result<Value> {
    let output = cmd.output().context("failed to spawn adapter command")?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        let stdout = String::from_utf8_lossy(&output.stdout);
        anyhow::bail!(
            "adapter command failed with status {}: {}\n{}",
            output.status,
            stdout.trim(),
            stderr.trim()
        );
    }
    let stdout = String::from_utf8(output.stdout).context("adapter stdout was not utf-8")?;
    serde_json::from_str(stdout.trim()).context("adapter stdout was not valid json")
}

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn now_iso_string() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    chrono_like_iso(now)
}

fn chrono_like_iso(epoch_seconds: u64) -> String {
    // Minimal UTC ISO-8601 formatter without adding a new dependency to the top-level crate.
    use std::fmt::Write as _;

    let seconds_per_day = 86_400u64;
    let days = epoch_seconds / seconds_per_day;
    let seconds_of_day = epoch_seconds % seconds_per_day;

    let (year, month, day) = civil_from_days(days as i64);
    let hour = seconds_of_day / 3_600;
    let minute = (seconds_of_day % 3_600) / 60;
    let second = seconds_of_day % 60;

    let mut output = String::with_capacity(20);
    let _ = write!(
        output,
        "{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}Z"
    );
    output
}

fn civil_from_days(days_since_unix_epoch: i64) -> (i64, i64, i64) {
    let z = days_since_unix_epoch + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = mp + if mp < 10 { 3 } else { -9 };
    let year = y + if m <= 2 { 1 } else { 0 };
    (year, m, d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn unique_test_db_path(prefix: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "{prefix}-{}.db",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ))
    }

    #[test]
    fn queue_tasks_round_trip_through_channel_store() {
        let root = std::env::temp_dir().join(format!(
            "ctox-queue-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("failed to create temp test root");

        let created = create_queue_task(
            &root,
            QueueTaskCreateRequest {
                title: "queue smoke".to_string(),
                prompt: "Inspect the queue task round-trip.".to_string(),
                thread_key: "queue/test".to_string(),
                priority: "high".to_string(),
                suggested_skill: Some("queue-orchestrator".to_string()),
                parent_message_key: None,
            },
        )
        .expect("failed to create queue task");
        assert_eq!(created.route_status, "pending");
        assert_eq!(created.priority, "high");
        assert_eq!(
            created.suggested_skill.as_deref(),
            Some("queue-orchestrator")
        );

        let updated = update_queue_task(
            &root,
            QueueTaskUpdateRequest {
                message_key: created.message_key.clone(),
                priority: Some("urgent".to_string()),
                route_status: Some("blocked".to_string()),
                status_note: Some("waiting for owner".to_string()),
                ..Default::default()
            },
        )
        .expect("failed to update queue task");
        assert_eq!(updated.route_status, "blocked");
        assert_eq!(updated.priority, "urgent");
        assert_eq!(updated.status_note.as_deref(), Some("waiting for owner"));

        let loaded = load_queue_task(&root, &created.message_key)
            .expect("failed to load queue task")
            .expect("queue task missing after update");
        assert_eq!(loaded.message_key, created.message_key);
        assert_eq!(loaded.route_status, "blocked");

        let listed = list_queue_tasks(&root, &["blocked".to_string()], 10)
            .expect("failed to list blocked queue tasks");
        assert_eq!(listed.len(), 1);
        assert_eq!(listed[0].message_key, created.message_key);

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn resolve_outbound_subject_reuses_existing_thread_subject() {
        let db_path = unique_test_db_path("ctox-channel-subject-reuse");
        let mut conn = open_channel_db(&db_path).expect("failed to open db");
        upsert_communication_message(
            &mut conn,
            UpsertMessage {
                message_key: "msg-1",
                channel: "email",
                account_key: "email:test@example.com",
                thread_key: "email/thread-1",
                remote_id: "remote-1",
                direction: "inbound",
                folder_hint: "INBOX",
                sender_display: "Owner",
                sender_address: "owner@example.com",
                recipient_addresses_json: "[]",
                cc_addresses_json: "[]",
                bcc_addresses_json: "[]",
                subject: "Existing subject",
                preview: "preview",
                body_text: "body",
                body_html: "",
                raw_payload_ref: "",
                trust_level: "owner_verified",
                status: "received",
                seen: false,
                has_attachments: false,
                external_created_at: "2026-03-26T10:00:00Z",
                observed_at: "2026-03-26T10:00:00Z",
                metadata_json: "{}",
            },
        )
        .expect("failed to upsert message");
        refresh_thread(&mut conn, "email/thread-1").expect("failed to refresh thread");

        let resolved = resolve_outbound_subject(
            &conn,
            ChannelSendRequest {
                channel: "email".to_string(),
                account_key: "email:test@example.com".to_string(),
                thread_key: "email/thread-1".to_string(),
                body: "reply".to_string(),
                subject: String::new(),
                to: vec!["owner@example.com".to_string()],
                cc: Vec::new(),
                sender_display: None,
                sender_address: None,
                send_voice: false,
            },
        )
        .expect("failed to resolve subject");
        assert_eq!(resolved.subject, "Existing subject");

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn resolve_outbound_subject_rejects_missing_email_subject() {
        let db_path = unique_test_db_path("ctox-channel-subject-missing");
        let conn = open_channel_db(&db_path).expect("failed to open db");
        let error = resolve_outbound_subject(
            &conn,
            ChannelSendRequest {
                channel: "email".to_string(),
                account_key: "email:test@example.com".to_string(),
                thread_key: "email/thread-2".to_string(),
                body: "reply".to_string(),
                subject: "(no subject)".to_string(),
                to: vec!["owner@example.com".to_string()],
                cc: Vec::new(),
                sender_display: None,
                sender_address: None,
                send_voice: false,
            },
        )
        .expect_err("missing email subject should fail");
        assert!(
            error
                .to_string()
                .contains("email send requires a real subject"),
            "unexpected error: {error}"
        );

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn system_probe_inbound_messages_are_marked_handled() {
        let db_path = unique_test_db_path("ctox-channel-system-probe");
        let mut conn = open_channel_db(&db_path).expect("failed to open db");
        upsert_communication_message(
            &mut conn,
            UpsertMessage {
                message_key: "probe-1",
                channel: "email",
                account_key: "email:test@example.com",
                thread_key: "email/thread-self-test",
                remote_id: "remote-probe-1",
                direction: "inbound",
                folder_hint: "INBOX",
                sender_display: "CTOX",
                sender_address: "test@example.com",
                recipient_addresses_json: "[]",
                cc_addresses_json: "[]",
                bcc_addresses_json: "[]",
                subject: "[CTOX mail self-test] 2026-03-26T10:00:00Z",
                preview: "CTOX self-test",
                body_text: "CTOX self-test <abc@example.com>",
                body_html: "",
                raw_payload_ref: "",
                trust_level: "system_probe",
                status: "self_test_received",
                seen: false,
                has_attachments: false,
                external_created_at: "2026-03-26T10:00:00Z",
                observed_at: "2026-03-26T10:00:00Z",
                metadata_json: "{\"technicalSelfTest\":true}",
            },
        )
        .expect("failed to insert system probe");
        ensure_routing_rows_for_inbound(&conn).expect("failed to backfill routing rows");
        let route_status: String = conn
            .query_row(
                "SELECT route_status FROM communication_routing_state WHERE message_key = ?1",
                params!["probe-1"],
                |row| row.get(0),
            )
            .expect("missing routing row");
        assert_eq!(route_status, "handled");

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn system_probe_routing_heals_existing_non_handled_rows() {
        let db_path = unique_test_db_path("ctox-channel-system-probe-heal");
        let mut conn = open_channel_db(&db_path).expect("failed to open db");
        upsert_communication_message(
            &mut conn,
            UpsertMessage {
                message_key: "probe-2",
                channel: "email",
                account_key: "email:test@example.com",
                thread_key: "email/thread-self-test-2",
                remote_id: "remote-probe-2",
                direction: "inbound",
                folder_hint: "INBOX",
                sender_display: "CTOX",
                sender_address: "test@example.com",
                recipient_addresses_json: "[]",
                cc_addresses_json: "[]",
                bcc_addresses_json: "[]",
                subject: "[CTOX mail self-test] 2026-03-26T10:10:00Z",
                preview: "CTOX self-test",
                body_text: "CTOX self-test <def@example.com>",
                body_html: "",
                raw_payload_ref: "",
                trust_level: "system_probe",
                status: "self_test_received",
                seen: false,
                has_attachments: false,
                external_created_at: "2026-03-26T10:10:00Z",
                observed_at: "2026-03-26T10:10:00Z",
                metadata_json: "{\"technicalSelfTest\":true}",
            },
        )
        .expect("failed to insert system probe");
        conn.execute(
            "INSERT INTO communication_routing_state (message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at) VALUES (?1, 'blocked_sender', NULL, NULL, NULL, 'legacy', ?2)",
            params!["probe-2", "2026-03-26T10:10:01Z"],
        )
        .expect("failed to seed legacy routing row");
        ensure_routing_rows_for_inbound(&conn).expect("failed to normalize routing rows");
        let route_status: String = conn
            .query_row(
                "SELECT route_status FROM communication_routing_state WHERE message_key = ?1",
                params!["probe-2"],
                |row| row.get(0),
            )
            .expect("missing routing row");
        assert_eq!(route_status, "handled");

        let _ = fs::remove_file(&db_path);
    }

    #[test]
    fn channel_history_and_search_can_reconstruct_related_messages() {
        let db_path = unique_test_db_path("ctox-channel-search");
        let mut conn = open_channel_db(&db_path).expect("failed to open db");
        for (message_key, thread_key, sender_address, subject, preview, body_text, created_at) in [
            (
                "msg-1",
                "email/thread-a",
                "owner@metric-space.ai",
                "Nextcloud blocked",
                "Need endpoint",
                "Nextcloud is blocked on missing endpoint and credentials.",
                "2026-03-26T10:00:00Z",
            ),
            (
                "msg-2",
                "email/thread-a",
                "ctox@metric-space.ai",
                "Nextcloud blocked",
                "Asked for endpoint",
                "I asked for NEXTCLOUD_URL and credentials.",
                "2026-03-26T10:05:00Z",
            ),
            (
                "msg-3",
                "email/thread-b",
                "owner@metric-space.ai",
                "Redis recovered",
                "Rotated secret",
                "The Redis password was rotated and the service is healthy now.",
                "2026-03-26T11:00:00Z",
            ),
        ] {
            upsert_communication_message(
                &mut conn,
                UpsertMessage {
                    message_key,
                    channel: "email",
                    account_key: "email:cto1@metric-space.ai",
                    thread_key,
                    remote_id: message_key,
                    direction: "inbound",
                    folder_hint: "INBOX",
                    sender_display: "Owner",
                    sender_address,
                    recipient_addresses_json: "[]",
                    cc_addresses_json: "[]",
                    bcc_addresses_json: "[]",
                    subject,
                    preview,
                    body_text,
                    body_html: "",
                    raw_payload_ref: "",
                    trust_level: "owner_verified",
                    status: "received",
                    seen: false,
                    has_attachments: false,
                    external_created_at: created_at,
                    observed_at: created_at,
                    metadata_json: "{}",
                },
            )
            .expect("failed to insert communication message");
        }

        let thread_history =
            list_thread_messages(&conn, "email/thread-a", 10).expect("failed to load history");
        assert_eq!(thread_history.len(), 2);
        assert_eq!(thread_history[0].message_key, "msg-2");

        let search =
            search_messages(&conn, "nextcloud endpoint", Some("email"), None, 10).expect("failed to search");
        assert_eq!(search.len(), 2);
        assert!(search.iter().all(|item| item.thread_key == "email/thread-a"));

        let sender_search = search_messages(
            &conn,
            "redis",
            Some("email"),
            Some("owner@metric-space.ai"),
            10,
        )
        .expect("failed to search sender-scoped messages");
        assert_eq!(sender_search.len(), 1);
        assert_eq!(sender_search[0].message_key, "msg-3");

        let _ = fs::remove_file(&db_path);
    }
}

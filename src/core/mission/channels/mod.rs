mod business_os_projection;
pub(crate) use business_os_projection::communication_intake_source_stamp;
use business_os_projection::non_negative_i64_to_usize;
pub use business_os_projection::{
    disconnect_communication_account_for_business_os, export_jami_archive_for_business_os,
    list_communication_accounts_for_business_os, pull_communication_accounts_for_business_os,
    pull_communication_accounts_for_business_os_after, pull_communication_messages_for_business_os,
    pull_communication_messages_for_business_os_after, pull_communication_record_for_business_os,
    pull_communication_threads_for_business_os, pull_communication_threads_for_business_os_after,
    read_pairing_state_for_business_os, save_channel_settings_for_business_os,
    start_pairing_for_business_os, sync_channel_for_business_os, test_channel_for_business_os,
};

use anyhow::Context;
use anyhow::Result;
use chrono::DateTime;
use chrono::Duration;
use chrono::Utc;
use qrcode::types::Color as QrColor;
use qrcode::QrCode;
use rusqlite::params;
use rusqlite::params_from_iter;
use rusqlite::types::Value as SqlValue;
use rusqlite::Connection;
use rusqlite::OpenFlags;
use rusqlite::OptionalExtension;
use rusqlite::Transaction;
use serde::Deserialize;
use serde::Serialize;
use serde_json::json;
use serde_json::Value;
use sha2::Digest;
use sha2::Sha256;
use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::collections::HashSet;
use std::fs;
#[cfg(unix)]
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::MutexGuard;
use std::sync::OnceLock;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::communication::adapters as communication_adapters;
use crate::communication::adapters::CommunicationTransportAdapter;
use crate::communication::gateway as communication_gateway;
use crate::mission::review::HoldReason;
use crate::secrets;
use crate::service::core_state_machine::{
    CoreEntityType, CoreEvent, CoreEvidenceRefs, CoreState, CoreTransitionRequest, RuntimeLane,
};
use crate::service::core_transition_guard::{
    enforce_core_spawn, enforce_core_spawn_in_transaction, enforce_core_transition,
    ensure_core_transition_guard_schema, evaluate_core_spawn, CoreSpawnProof, CoreSpawnRequest,
};
use crate::service::harness_flow::{
    record_harness_flow_event_lossy, RecordHarnessFlowEventRequest,
};

const DEFAULT_TAKE_LIMIT: usize = 10;
const QUEUE_CHANNEL_NAME: &str = "queue";
const QUEUE_ACCOUNT_KEY: &str = "queue:system";
const QUEUE_ACCOUNT_ADDRESS: &str = "ctox queue";
const QUEUE_PROVIDER: &str = "system";
const QUEUE_SENDER_DISPLAY: &str = "CTOX queue";
const QUEUE_SENDER_ADDRESS: &str = "queue:system";
static REVIEWED_FOUNDER_SEND_LOCK: OnceLock<Mutex<()>> = OnceLock::new();
static CHANNEL_SCHEMA_READY: OnceLock<Mutex<HashSet<ChannelSchemaCacheKey>>> = OnceLock::new();
static CHANNEL_OPEN_ROUTING_READY: OnceLock<
    Mutex<BTreeMap<ChannelSchemaCacheKey, ChannelRoutingCacheStamp>>,
> = OnceLock::new();
static QUEUE_TASK_LIST_CACHE: OnceLock<
    Mutex<BTreeMap<QueueTaskListCacheKey, QueueTaskListCacheEntry>>,
> = OnceLock::new();
static QUEUE_TASK_COUNT_CACHE: OnceLock<
    Mutex<BTreeMap<QueueTaskCountCacheKey, QueueTaskCountCacheEntry>>,
> = OnceLock::new();

const QUEUE_TASK_LIST_CACHE_MAX_ENTRIES: usize = 256;
const QUEUE_TASK_COUNT_CACHE_MAX_ENTRIES: usize = 256;

type ChannelFileChangeStamp = (u64, u128);
type ChannelRoutingCacheStamp = (u64, u64, u64);

#[derive(Debug, Clone, PartialEq, Eq)]
enum QueueTaskListCacheStamp {
    ProjectionClock {
        database_exists: bool,
        clock_exists: bool,
        version: i64,
        message_count: usize,
        routing_count: usize,
        updated_at: String,
    },
    File {
        main: ChannelFileChangeStamp,
        wal: ChannelFileChangeStamp,
        journal: ChannelFileChangeStamp,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct QueueTaskListCacheKey {
    database: ChannelSchemaCacheKey,
    statuses: Vec<String>,
    limit: usize,
}

#[derive(Debug, Clone)]
struct QueueTaskListCacheEntry {
    stamp: QueueTaskListCacheStamp,
    tasks: Vec<QueueTaskView>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct QueueTaskCountCacheKey {
    database: ChannelSchemaCacheKey,
    statuses: Vec<String>,
}

#[derive(Debug, Clone)]
struct QueueTaskCountCacheEntry {
    stamp: QueueTaskListCacheStamp,
    count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct CommunicationIntakeSourceStamp {
    database_exists: bool,
    accounts_table_exists: bool,
    threads_table_exists: bool,
    messages_table_exists: bool,
    routing_table_exists: bool,
    projection_version: i64,
    account_count: usize,
    latest_account_updated_at_ms: i64,
    thread_count: usize,
    latest_thread_updated_at_ms: i64,
    message_count: usize,
    latest_message_updated_at_ms: i64,
    routing_count: usize,
    clock_updated_at: String,
    content_hash: String,
}

const COMMUNICATION_INTAKE_SOURCE_STAMP_SQL: &str = r#"
    SELECT
        version,
        account_count,
        thread_count,
        message_count,
        routing_count,
        updated_at
    FROM communication_projection_clock
    WHERE id = 1
"#;

#[cfg(test)]
static CHANNEL_SCHEMA_ENSURE_COUNTS: OnceLock<Mutex<BTreeMap<ChannelSchemaCacheKey, usize>>> =
    OnceLock::new();

#[cfg(unix)]
type ChannelSchemaCacheKey = (PathBuf, u64, u64);
#[cfg(not(unix))]
type ChannelSchemaCacheKey = PathBuf;

#[cfg(test)]
static CHANNEL_OPEN_ROUTING_ENSURE_COUNTS: OnceLock<Mutex<BTreeMap<ChannelSchemaCacheKey, usize>>> =
    OnceLock::new();

#[cfg(test)]
static QUEUE_TASK_LIST_CACHE_MISS_COUNTS: OnceLock<Mutex<BTreeMap<QueueTaskListCacheKey, usize>>> =
    OnceLock::new();

#[cfg(test)]
static QUEUE_TASK_COUNT_CACHE_MISS_COUNTS: OnceLock<
    Mutex<BTreeMap<QueueTaskCountCacheKey, usize>>,
> = OnceLock::new();
#[cfg(test)]
static CHANNEL_DB_OPEN_CALL_COUNTS: OnceLock<Mutex<BTreeMap<PathBuf, usize>>> = OnceLock::new();

#[derive(Debug, Clone, Serialize)]
pub struct QueueTaskView {
    pub message_key: String,
    pub thread_key: String,
    pub title: String,
    pub prompt: String,
    pub workspace_root: Option<String>,
    pub ticket_self_work_id: Option<String>,
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
pub(crate) struct BusinessCommandClaimRequest {
    pub command_id: String,
    pub idempotency_key: String,
    pub payload_hash: String,
    pub module: String,
    pub command_type: String,
    pub record_id: String,
    pub intent: Value,
    pub created_at_ms: i64,
}

#[derive(Debug, Clone)]
pub(crate) struct BusinessCommandQueueClaim {
    pub task: QueueTaskView,
    pub already_claimed: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct BusinessCommandControlClaim {
    pub disposition: &'static str,
    pub result: Option<Value>,
    pub terminal_status: Option<String>,
}

#[derive(Debug, Clone)]
pub(crate) struct BusinessCommandOutboxEvent {
    pub event_id: String,
    pub command_id: String,
    pub projection_version: i64,
    pub destination: String,
    pub event_type: String,
    pub attempts: u32,
}

#[derive(Debug, Clone)]
pub struct QueueTaskCreateRequest {
    pub title: String,
    pub prompt: String,
    pub thread_key: String,
    pub workspace_root: Option<String>,
    pub priority: String,
    pub suggested_skill: Option<String>,
    pub parent_message_key: Option<String>,
    pub extra_metadata: Option<Value>,
}

#[derive(Debug, Clone, Default)]
pub struct QueueTaskUpdateRequest {
    pub message_key: String,
    pub title: Option<String>,
    pub prompt: Option<String>,
    pub thread_key: Option<String>,
    pub workspace_root: Option<String>,
    pub clear_workspace_root: bool,
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
    pub founder_email_addresses: Vec<String>,
    pub founder_email_roles: Vec<String>,
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

    let setting = |key: &str| -> String {
        settings
            .get(key)
            .map(|value| value.trim().to_string())
            .unwrap_or_default()
    };
    let first_setting = |keys: &[&str]| -> String {
        keys.iter()
            .find_map(|key| {
                settings
                    .get(*key)
                    .map(|value| value.trim())
                    .filter(|value| !value.is_empty())
                    .map(str::to_string)
            })
            .unwrap_or_default()
    };
    let split_list = |raw: &str| -> Vec<String> {
        raw.split(|ch| matches!(ch, ',' | ';' | '\n'))
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .map(str::to_string)
            .collect()
    };
    let first_list = |keys: &[&str]| -> Vec<String> {
        keys.iter()
            .find_map(|key| {
                settings
                    .get(*key)
                    .map(|value| split_list(value))
                    .filter(|values| !values.is_empty())
            })
            .unwrap_or_default()
    };
    let mut ensure_chat_account = |channel: &str,
                                   provider: &str,
                                   id: String,
                                   address: String,
                                   profile_json: Value|
     -> Result<()> {
        if id.trim().is_empty() && address.trim().is_empty() {
            return Ok(());
        }
        let suffix = if id.trim().is_empty() {
            stable_digest(&address)
        } else {
            id.trim().to_string()
        };
        ensure_account(
            &mut conn,
            &format!("{channel}:{suffix}"),
            channel,
            if address.trim().is_empty() {
                &suffix
            } else {
                address.trim()
            },
            provider,
            profile_json,
        )
    };

    if !first_setting(&[
        "CTO_SLACK_BOT_TOKEN",
        "CTO_SLACK_WORKSPACE_ID",
        "CTO_SLACK_CHANNEL_ID",
    ])
    .is_empty()
    {
        let workspace_id = setting("CTO_SLACK_WORKSPACE_ID");
        let bot_user_id = setting("CTO_SLACK_BOT_USER_ID");
        ensure_chat_account(
            "slack",
            "slack-web-api",
            first_setting(&["CTO_SLACK_BOT_USER_ID", "CTO_SLACK_WORKSPACE_ID"]),
            if bot_user_id.is_empty() {
                workspace_id.clone()
            } else {
                bot_user_id.clone()
            },
            json!({
                "workspaceId": workspace_id,
                "botUserId": bot_user_id,
                "channelIds": first_list(&["CTO_SLACK_CHANNEL_IDS", "CTO_SLACK_CHANNEL_ID"]),
                "apiBaseUrl": setting("CTO_SLACK_API_BASE_URL"),
            }),
        )?;
    }

    if !first_setting(&[
        "CTO_DISCORD_BOT_TOKEN",
        "CTO_DISCORD_APPLICATION_ID",
        "CTO_DISCORD_CHANNEL_ID",
    ])
    .is_empty()
    {
        let application_id = setting("CTO_DISCORD_APPLICATION_ID");
        ensure_chat_account(
            "discord",
            "discord-rest",
            first_setting(&["CTO_DISCORD_APPLICATION_ID", "CTO_DISCORD_BOT_USER_ID"]),
            if application_id.is_empty() {
                setting("CTO_DISCORD_BOT_USER_ID")
            } else {
                application_id.clone()
            },
            json!({
                "applicationId": application_id,
                "botUserId": setting("CTO_DISCORD_BOT_USER_ID"),
                "guildIds": first_list(&["CTO_DISCORD_GUILD_IDS", "CTO_DISCORD_GUILD_ID"]),
                "channelIds": first_list(&["CTO_DISCORD_CHANNEL_IDS", "CTO_DISCORD_CHANNEL_ID"]),
                "apiBaseUrl": setting("CTO_DISCORD_API_BASE_URL"),
            }),
        )?;
    }

    if !first_setting(&["CTO_TELEGRAM_BOT_TOKEN", "CTO_TELEGRAM_BOT_USERNAME"]).is_empty() {
        let bot_username = setting("CTO_TELEGRAM_BOT_USERNAME");
        ensure_chat_account(
            "telegram",
            "telegram-bot-api",
            bot_username.clone(),
            bot_username.clone(),
            json!({
                "botUsername": bot_username,
                "chatIds": first_list(&["CTO_TELEGRAM_CHAT_IDS", "CTO_TELEGRAM_CHAT_ID"]),
                "apiBaseUrl": setting("CTO_TELEGRAM_API_BASE_URL"),
            }),
        )?;
    }

    if !first_setting(&["CTO_MATRIX_HOMESERVER_URL", "CTO_MATRIX_USER_ID"]).is_empty() {
        let user_id = setting("CTO_MATRIX_USER_ID");
        ensure_chat_account(
            "matrix",
            "matrix-client-server",
            user_id.clone(),
            user_id.clone(),
            json!({
                "homeserverUrl": setting("CTO_MATRIX_HOMESERVER_URL"),
                "userId": user_id,
                "roomIds": first_list(&["CTO_MATRIX_ROOM_IDS", "CTO_MATRIX_ROOM_ID"]),
            }),
        )?;
    }

    if !first_setting(&[
        "CTO_MATTERMOST_SERVER_URL",
        "CTO_MATTERMOST_BOT_USER_ID",
        "CTO_MATTERMOST_CHANNEL_ID",
    ])
    .is_empty()
    {
        let bot_user_id = setting("CTO_MATTERMOST_BOT_USER_ID");
        ensure_chat_account(
            "mattermost",
            "mattermost-api-v4",
            first_setting(&["CTO_MATTERMOST_BOT_USER_ID", "CTO_MATTERMOST_SERVER_URL"]),
            if bot_user_id.is_empty() {
                setting("CTO_MATTERMOST_SERVER_URL")
            } else {
                bot_user_id.clone()
            },
            json!({
                "serverUrl": setting("CTO_MATTERMOST_SERVER_URL"),
                "botUserId": bot_user_id,
                "teamId": setting("CTO_MATTERMOST_TEAM_ID"),
                "channelIds": first_list(&["CTO_MATTERMOST_CHANNEL_IDS", "CTO_MATTERMOST_CHANNEL_ID"]),
            }),
        )?;
    }

    if !first_setting(&[
        "CTO_ZULIP_REALM_URL",
        "CTO_ZULIP_BOT_EMAIL",
        "CTO_ZULIP_EMAIL",
    ])
    .is_empty()
    {
        let bot_email = first_setting(&["CTO_ZULIP_BOT_EMAIL", "CTO_ZULIP_EMAIL"]);
        ensure_chat_account(
            "zulip",
            "zulip-rest-api",
            bot_email.clone(),
            bot_email.clone(),
            json!({
                "realmUrl": setting("CTO_ZULIP_REALM_URL"),
                "botEmail": bot_email,
                "streams": first_list(&["CTO_ZULIP_STREAMS", "CTO_ZULIP_STREAM"]),
                "topic": setting("CTO_ZULIP_TOPIC"),
            }),
        )?;
    }

    if !first_setting(&[
        "CTO_GOOGLE_CHAT_USER",
        "CTO_GOOGLE_CHAT_APP_ID",
        "CTO_GOOGLE_CHAT_SPACE_NAME",
    ])
    .is_empty()
    {
        let account_id = first_setting(&["CTO_GOOGLE_CHAT_USER", "CTO_GOOGLE_CHAT_APP_ID"]);
        ensure_chat_account(
            "google_chat",
            "google-chat-api",
            account_id.clone(),
            account_id.clone(),
            json!({
                "user": setting("CTO_GOOGLE_CHAT_USER"),
                "appId": setting("CTO_GOOGLE_CHAT_APP_ID"),
                "spaceNames": first_list(&["CTO_GOOGLE_CHAT_SPACE_NAMES", "CTO_GOOGLE_CHAT_SPACE_NAME"]),
                "apiBaseUrl": setting("CTO_GOOGLE_CHAT_API_BASE_URL"),
            }),
        )?;
    }

    Ok(())
}

pub fn merge_owner_profile_settings(
    root: &Path,
    settings: &mut BTreeMap<String, String>,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let mut stmt = conn.prepare(
        r#"
        SELECT owner_key, metadata_json
        FROM owner_profiles
        ORDER BY owner_key ASC
        "#,
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;

    let mut founder_emails = parse_founder_email_addresses(settings)
        .into_iter()
        .collect::<BTreeSet<_>>();
    let mut founder_roles = parse_founder_email_roles(settings);
    let mut admin_policies = parse_admin_email_policies(settings)
        .into_iter()
        .map(|entry| (entry.email, entry.can_sudo))
        .collect::<BTreeMap<_, _>>();

    for row in rows {
        let (owner_key, metadata_json) = row?;
        let metadata = serde_json::from_str::<Value>(&metadata_json).unwrap_or(Value::Null);
        let email = metadata
            .get("email")
            .and_then(Value::as_str)
            .map(normalize_email_address)
            .filter(|value| !value.is_empty())
            .or_else(|| {
                let normalized = normalize_email_address(&owner_key);
                normalized.contains('@').then_some(normalized)
            });
        let Some(email) = email else {
            continue;
        };
        let role = metadata
            .get("role")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        match role.as_str() {
            "owner" => {
                settings
                    .entry("CTOX_OWNER_EMAIL_ADDRESS".to_string())
                    .or_insert(email);
            }
            "founder" => {
                founder_emails.insert(email.clone());
                let role_title = metadata
                    .get("role_title")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or("Founder");
                founder_roles
                    .entry(email)
                    .or_insert_with(|| role_title.to_string());
            }
            "admin" => {
                let can_sudo = metadata
                    .get("allow_sudo_actions")
                    .and_then(Value::as_bool)
                    .unwrap_or(false);
                admin_policies.entry(email).or_insert(can_sudo);
            }
            _ => {}
        }
    }

    if !founder_emails.is_empty() {
        settings.insert(
            "CTOX_FOUNDER_EMAIL_ADDRESSES".to_string(),
            founder_emails.into_iter().collect::<Vec<_>>().join(","),
        );
    }
    if !founder_roles.is_empty() {
        settings.insert(
            "CTOX_FOUNDER_EMAIL_ROLES".to_string(),
            founder_roles
                .into_iter()
                .map(|(email, role)| format!("{email}={role}"))
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    if !admin_policies.is_empty() {
        settings.insert(
            "CTOX_EMAIL_ADMIN_POLICIES".to_string(),
            admin_policies
                .into_iter()
                .map(|(email, can_sudo)| {
                    if can_sudo {
                        format!("{email}=sudo")
                    } else {
                        email
                    }
                })
                .collect::<Vec<_>>()
                .join(","),
        );
    }
    Ok(())
}

fn runtime_settings_with_owner_profiles(
    root: &Path,
    kind: communication_gateway::CommunicationAdapterKind,
) -> BTreeMap<String, String> {
    let mut settings = communication_gateway::runtime_settings_from_root(root, kind);
    let _ = merge_owner_profile_settings(root, &mut settings);
    settings
}

pub fn ensure_store(root: &Path) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let _conn = open_channel_db(&db_path)?;
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
            "teams" => {
                let parsed =
                    serde_json::from_str::<Value>(&profile_json).unwrap_or_else(|_| json!({}));
                let bot_id = parsed
                    .get("botId")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or(address.trim());
                if !bot_id.is_empty() {
                    channels.insert(format!("- teams: {}", bot_id));
                }
            }
            "whatsapp" => {
                let parsed =
                    serde_json::from_str::<Value>(&profile_json).unwrap_or_else(|_| json!({}));
                let jid = parsed
                    .get("jid")
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .unwrap_or(address.trim());
                if !jid.is_empty() {
                    channels.insert(format!("- whatsapp: {}", jid));
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
        founder_email_addresses: parse_founder_email_addresses(settings),
        founder_email_roles: founder_email_role_summaries(settings),
        allowed_email_domain: normalized_allowed_email_domain(settings),
        admin_email_policies: admin_email_policy_summaries(settings),
        channels: channels.into_iter().collect(),
        preferred_channel: settings
            .get("CTOX_OWNER_PREFERRED_CHANNEL")
            .map(|value| value.trim().to_string())
            .filter(|value| !value.is_empty()),
    })
}

/// Whether the inbound message metadata carries the structured
/// "auto-submitted" marker we extract from RFC 3834 / Outlook headers
/// at IMAP/Graph ingestion time.
///
/// We deliberately do NOT inspect subject lines or body text here:
/// language- and template-specific scraping belongs in skills, not in
/// the core. This check looks only at JSON fields written by the
/// inbound parser.
pub fn metadata_marks_auto_submitted(metadata: &Value) -> bool {
    let direct = metadata
        .get("autoSubmitted")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if direct {
        return true;
    }
    let suppress = metadata
        .get("autoResponseSuppress")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if suppress {
        return true;
    }
    // Defense-in-depth: when the inbound parser captured the raw
    // header value but failed to populate the boolean (older row
    // shape), still honour an `auto-replied`/`auto-generated`/
    // `auto-notified` token. We compare structured tokens, not
    // free-form strings.
    if let Some(value) = metadata.get("autoSubmittedValue").and_then(Value::as_str) {
        let token = value
            .split(';')
            .next()
            .unwrap_or("")
            .trim()
            .to_ascii_lowercase();
        if !token.is_empty() && token != "no" {
            return true;
        }
    }
    false
}

pub fn reclassify_historical_auto_submitted_inbounds(root: &Path) -> Result<usize> {
    #[derive(Debug)]
    struct Candidate {
        message_key: String,
        subject: String,
        sender_address: String,
        body_text: String,
        metadata: Value,
    }

    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let mut stmt = conn.prepare(
        r#"
        SELECT
            m.message_key,
            m.subject,
            m.sender_address,
            m.body_text,
            m.metadata_json
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.direction = 'inbound'
          AND m.status = 'received'
          AND m.channel = 'email'
          AND COALESCE(r.route_status, 'pending') IN ('pending','leased','failed','review_rework','handled')
          AND NOT EXISTS (
              SELECT 1
              FROM communication_founder_reply_reviews review
              WHERE review.inbound_message_key = m.message_key
                AND review.terminal_no_send = 1
          )
        "#,
    )?;
    let candidates = stmt
        .query_map([], |row| {
            let metadata_raw: String = row.get(4)?;
            Ok(Candidate {
                message_key: row.get(0)?,
                subject: row.get(1)?,
                sender_address: row.get(2)?,
                body_text: row.get(3)?,
                metadata: serde_json::from_str(&metadata_raw).unwrap_or_else(|_| json!({})),
            })
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;

    let mut reclassified = 0usize;
    for mut candidate in candidates {
        if metadata_marks_auto_submitted(&candidate.metadata) {
            continue;
        }
        let Some(reason) = historical_auto_submitted_reason(
            &candidate.subject,
            &candidate.sender_address,
            &candidate.body_text,
        ) else {
            continue;
        };
        let now = now_iso_string();
        if let Some(object) = candidate.metadata.as_object_mut() {
            object.insert("autoSubmitted".to_string(), Value::Bool(true));
            object.insert(
                "autoSubmittedValue".to_string(),
                Value::String("historical-reclassifier".to_string()),
            );
            object.insert("terminalNoSend".to_string(), Value::Bool(true));
            object.insert(
                "terminalNoSendReason".to_string(),
                Value::String(reason.clone()),
            );
            object.insert("reclassifiedAt".to_string(), Value::String(now.clone()));
        }
        conn.execute(
            r#"
            UPDATE communication_messages
            SET metadata_json = ?2
            WHERE message_key = ?1
            "#,
            params![
                candidate.message_key,
                serde_json::to_string(&candidate.metadata)?
            ],
        )?;
        record_terminal_no_send_verdict(
            root,
            &candidate.message_key,
            "boot-reclassifier",
            &reason,
        )?;
        let previous_route_status = current_queue_route_status(&conn, &candidate.message_key)?;
        enforce_queue_route_status_transition(
            &conn,
            &candidate.message_key,
            &previous_route_status,
            "handled",
            "ctox-boot-reclassifier",
            "mark_historical_auto_submitted_inbound_handled",
        )?;
        conn.execute(
            r#"
            INSERT INTO communication_routing_state (
                message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at
            )
            VALUES (?1, 'handled', NULL, NULL, ?2, NULL, ?2)
            ON CONFLICT(message_key) DO UPDATE SET
                route_status='handled',
                lease_owner=NULL,
                leased_at=NULL,
                acked_at=?2,
                last_error=NULL,
                updated_at=?2
            "#,
            params![candidate.message_key, now],
        )?;
        reclassified += 1;
    }
    Ok(reclassified)
}

fn historical_auto_submitted_reason(
    subject: &str,
    sender_address: &str,
    body_text: &str,
) -> Option<String> {
    let subject = subject.trim().to_ascii_lowercase();
    if [
        "automatische antwort:",
        "auto-reply:",
        "out of office:",
        "automatic reply:",
    ]
    .iter()
    .any(|prefix| subject.starts_with(prefix))
    {
        return Some("historical auto-reply subject: terminal NO-SEND".to_string());
    }

    let sender = normalize_email_address(sender_address);
    let local_part = sender.split('@').next().unwrap_or("");
    if matches!(
        local_part,
        "noreply" | "no-reply" | "donotreply" | "do-not-reply" | "notification" | "notifications"
    ) {
        return Some("historical notification sender: terminal NO-SEND".to_string());
    }

    if body_is_only_teams_meeting_link(body_text) {
        return Some(
            "historical Teams meeting-link notification without human content: terminal NO-SEND"
                .to_string(),
        );
    }
    None
}

fn body_is_only_teams_meeting_link(body_text: &str) -> bool {
    let lowered = body_text.to_ascii_lowercase();
    if !(lowered.contains("teams.microsoft.com/l/meetup-join")
        || lowered.contains("teams.live.com/meet")
        || lowered.contains("join.microsoft.com/meet"))
    {
        return false;
    }
    let mut remainder = lowered.as_str();
    let mut cleaned = String::new();
    while let Some(start) = remainder.find("http") {
        cleaned.push_str(&remainder[..start]);
        let after_start = &remainder[start..];
        let end = after_start
            .find(char::is_whitespace)
            .unwrap_or(after_start.len());
        remainder = &after_start[end..];
    }
    cleaned.push_str(remainder);
    for phrase in [
        "microsoft teams",
        "join the meeting",
        "meeting id",
        "passcode",
        "dial in",
        "privacy and security",
        "learn more",
        "need help",
        "besprechungs-id",
        "kenncode",
        "an besprechung teilnehmen",
        "teilnehmen",
    ] {
        cleaned = cleaned.replace(phrase, " ");
    }
    let meaningful = cleaned
        .chars()
        .filter(|ch| ch.is_alphanumeric())
        .collect::<String>();
    meaningful.len() <= 40
}

/// Terminal route states that are sticky against further re-routing.
/// Once an inbound message is acked into one of these, the service
/// loop must NOT pull it back into `review_rework` or any other
/// non-terminal state. New work for the same thread must arrive via a
/// fresh inbound message (with its own message_key).
pub fn route_status_is_terminal(route_status: &str) -> bool {
    matches!(
        route_status,
        "handled" | "cancelled" | "failed" | "completed"
    )
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
    let founder_emails = parse_founder_email_addresses(settings);
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

    if founder_emails
        .iter()
        .any(|email| email == &normalized_email)
    {
        return EmailSenderPolicy {
            normalized_email,
            role: "founder".to_string(),
            allowed: true,
            allow_admin_actions: true,
            allow_sudo_actions: false,
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
        block_reason: Some(
            "sender is outside the configured founder/owner/admin list and allowed employee email domain"
                .to_string(),
        ),
    }
}

/// F4: snapshot of the founder/owner outbound pipeline for a single thread.
/// Joins:
/// - `mission_states`        — current mission status, agent_failure_count
/// - `messages`              — agent attempts and their structured outcomes
/// - `communication_founder_reply_reviews` — review and approval records
/// - `communication_messages` (outbound rows, plus their routing state) —
///   actual send attempts and their delivery state
///
/// Output is flat JSON shaped for operator consumption, intentionally
/// avoiding internal-only field names where they would leak past CTOX.
#[derive(Debug, Clone, Serialize)]
pub struct PipelineStatusReport {
    pub thread_key: Option<String>,
    pub founder_outbound_intent: bool,
    pub agent_attempts: Vec<PipelineAgentAttempt>,
    pub review_runs: Vec<PipelineReviewRun>,
    pub approval_records: Vec<PipelineApprovalRecord>,
    pub send_attempts: Vec<PipelineSendAttempt>,
    pub current_mission_status: String,
    pub agent_failure_count: i64,
    /// Iteration counter for the lightweight rewrite-only review path
    /// (per-mission, reset on approval). Surfaced so operators can see
    /// when a thread is bouncing in the body-fix loop versus the heavy
    /// rework path.
    pub rewrite_iteration_count: i64,
    /// Iteration counter for the heavy rework path. Derived from the
    /// stored `agent_failure_count` because rework continuations inherit
    /// the agent-failure backoff machinery; this duplication keeps the
    /// pipeline-status surface self-describing without changing the
    /// underlying schema.
    pub rework_iteration_count: i64,
    /// Most recent disposition the dispatcher chose. One of `Approved`,
    /// `RewriteOnly`, `RequeueInternalWork`, `None`. Computed from the latest
    /// review run / mission status, so it stays accurate without an
    /// extra column.
    pub current_disposition: String,
    pub last_error: Option<String>,
    /// Recent governance events from the strategic-directive owner-authority
    /// gate that touched this thread. Surfaces both permitted and blocked
    /// inbound-mail-driven mutations so operators can see whether the
    /// authority gate fired (and how) for the conversation. Filtered by
    /// `details.thread_key` or `details.conversation_id` so unrelated
    /// global authority events do not leak into the per-thread surface.
    pub strategic_directive_authority_events: Vec<StrategicDirectiveAuthorityEvent>,
}

#[derive(Debug, Clone, Serialize)]
pub struct StrategicDirectiveAuthorityEvent {
    pub event_id: String,
    pub mechanism_id: String,
    pub severity: String,
    pub created_at: String,
    pub sender_role: Option<String>,
    pub sender_address: Option<String>,
    pub directive_kind: Option<String>,
    pub attempted_status: Option<String>,
    pub action: Option<String>,
    pub triggered_by_message_key: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineAgentAttempt {
    pub turn_id: String,
    pub outcome: Option<String>,
    pub started_at: Option<String>,
    pub ended_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineReviewRun {
    pub approval_key: String,
    pub inbound_message_key: String,
    pub reviewer: String,
    pub review_summary: String,
    pub approved_at: String,
    pub sent_at: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineApprovalRecord {
    pub approval_key: String,
    pub action_digest: String,
    pub body_sha256: String,
    pub approved_at: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct PipelineSendAttempt {
    pub message_key: String,
    pub direction: String,
    pub subject: String,
    pub external_created_at: String,
    pub route_status: Option<String>,
    pub last_error: Option<String>,
}

pub(crate) fn pipeline_status(
    root: &Path,
    thread_key: Option<&str>,
    limit: usize,
) -> Result<PipelineStatusReport> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;

    // Agent attempts and reviews are scoped per-conversation_id derived
    // from the thread_key. If the operator did not supply one, we report
    // global state without per-thread review/send rows.
    let conversation_id = thread_key
        .map(|key| crate::execution::agent::turn_loop::conversation_id_for_thread_key(Some(key)));

    // Mission state for the conversation that owns this thread.
    let mission_state = if let Some(conv_id) = conversation_id {
        crate::lcm::LcmEngine::open(&db_path, crate::lcm::LcmConfig::default())
            .ok()
            .and_then(|engine| engine.stored_mission_state(conv_id).ok().flatten())
    } else {
        None
    };
    let (
        current_mission_status,
        agent_failure_count,
        rewrite_iteration_count,
        rework_iteration_count,
        last_error,
    ) = match &mission_state {
        Some(record) => (
            record.mission_status.clone(),
            record.agent_failure_count,
            record.rewrite_failure_count,
            record.agent_failure_count,
            record.deferred_reason.clone(),
        ),
        None => ("unknown".to_string(), 0, 0, 0, None),
    };

    // Agent attempts: most recent assistant rows for the conversation, in
    // reverse-chronological order, along with their structured outcome.
    let agent_attempts = if let Some(conv_id) = conversation_id {
        let mut stmt = conn.prepare(
            "SELECT message_id, agent_outcome, created_at
             FROM messages
             WHERE conversation_id = ?1 AND role = 'assistant'
             ORDER BY seq DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![conv_id, limit as i64], |row| {
            let id: i64 = row.get(0)?;
            let outcome: Option<String> = row.get(1)?;
            let ended: Option<String> = row.get(2)?;
            Ok(PipelineAgentAttempt {
                turn_id: format!("msg:{id}"),
                outcome,
                started_at: None,
                ended_at: ended,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

    // Review and approval records keyed on the inbound message belonging
    // to this thread. If thread_key is None, return all recent reviews.
    let mut review_runs = Vec::new();
    let mut approval_records = Vec::new();
    if let Some(thread) = thread_key {
        let mut stmt = conn.prepare(
            "SELECT r.approval_key, r.inbound_message_key, r.action_digest, r.body_sha256,
                    r.reviewer, r.review_summary, r.approved_at, r.sent_at
             FROM communication_founder_reply_reviews r
             JOIN communication_messages m ON m.message_key = r.inbound_message_key
             WHERE m.thread_key = ?1
             ORDER BY r.approved_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![thread, limit as i64], |row| {
            let approval_key: String = row.get(0)?;
            let inbound_message_key: String = row.get(1)?;
            let action_digest: String = row.get(2)?;
            let body_sha256: String = row.get(3)?;
            let reviewer: String = row.get(4)?;
            let review_summary: String = row.get(5)?;
            let approved_at: String = row.get(6)?;
            let sent_at: Option<String> = row.get(7)?;
            Ok((
                PipelineReviewRun {
                    approval_key: approval_key.clone(),
                    inbound_message_key,
                    reviewer,
                    review_summary,
                    approved_at: approved_at.clone(),
                    sent_at,
                },
                PipelineApprovalRecord {
                    approval_key,
                    action_digest,
                    body_sha256,
                    approved_at,
                },
            ))
        })?;
        for row in rows {
            let (review, approval) = row?;
            review_runs.push(review);
            approval_records.push(approval);
        }
    }

    // Send attempts: outbound communication_messages rows for this thread.
    let send_attempts = if let Some(thread) = thread_key {
        let mut stmt = conn.prepare(
            "SELECT m.message_key, m.direction, m.subject, m.external_created_at,
                    r.route_status, r.last_error
             FROM communication_messages m
             LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
             WHERE m.thread_key = ?1 AND m.direction = 'outbound'
             ORDER BY m.external_created_at DESC, m.observed_at DESC
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![thread, limit as i64], |row| {
            Ok(PipelineSendAttempt {
                message_key: row.get(0)?,
                direction: row.get(1)?,
                subject: row.get(2)?,
                external_created_at: row.get(3)?,
                route_status: row.get(4)?,
                last_error: row.get(5)?,
            })
        })?;
        rows.collect::<rusqlite::Result<Vec<_>>>()?
    } else {
        Vec::new()
    };

    // founder_outbound_intent is true if there's at least one approval
    // record for this thread (a reviewed founder send was prepared).
    let founder_outbound_intent = !approval_records.is_empty();

    // The dispatcher disposition is structural (no string scraping). We
    // derive it from the persisted state: an approval row implies the most
    // recent disposition was `Approved`; a non-zero rewrite_failure_count
    // implies the loop is in the lightweight rewrite path; a non-zero
    // agent_failure_count implies the heavy rework path. Otherwise the
    // pipeline has never produced a reviewed slice — `None`.
    let current_disposition = if !approval_records.is_empty() {
        "Approved".to_string()
    } else if rewrite_iteration_count > 0 {
        "RewriteOnly".to_string()
    } else if rework_iteration_count > 0 {
        "RequeueInternalWork".to_string()
    } else {
        "None".to_string()
    };

    // E (PR): per-thread strategic-directive authority audit trail. We
    // pull both the `_owner_authorised` and `_blocked_non_owner_sender`
    // events the strategy-mutation gate emits, and filter to those whose
    // structured details reference this thread (`thread_key` or
    // `conversation_id`). The default surface is the last `limit` such
    // events; if no thread was supplied we leave the list empty rather
    // than reporting global state, which matches how the surrounding
    // pipeline fields treat an absent thread_key.
    let strategic_directive_authority_events =
        load_strategic_directive_authority_events(&db_path, thread_key, conversation_id, limit)?;

    Ok(PipelineStatusReport {
        thread_key: thread_key.map(ToOwned::to_owned),
        founder_outbound_intent,
        agent_attempts,
        review_runs,
        approval_records,
        send_attempts,
        current_mission_status,
        agent_failure_count,
        rewrite_iteration_count,
        rework_iteration_count,
        current_disposition,
        last_error,
        strategic_directive_authority_events,
    })
}

fn load_strategic_directive_authority_events(
    db_path: &Path,
    thread_key: Option<&str>,
    conversation_id: Option<i64>,
    limit: usize,
) -> Result<Vec<StrategicDirectiveAuthorityEvent>> {
    if thread_key.is_none() && conversation_id.is_none() {
        return Ok(Vec::new());
    }
    // The governance schema is created lazily by the governance module; if
    // it does not exist yet, return an empty vec rather than erroring.
    let conn = Connection::open(db_path).with_context(|| {
        format!(
            "failed to open db {} for strategic-directive authority events",
            db_path.display()
        )
    })?;
    let exists: bool = conn
        .query_row(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='governance_events'",
            [],
            |row| row.get::<_, i64>(0),
        )
        .optional()?
        .is_some();
    if !exists {
        return Ok(Vec::new());
    }
    let mut stmt = conn.prepare(
        "SELECT event_id, mechanism_id, severity, details_json, created_at
         FROM governance_events
         WHERE mechanism_id IN (
             'strategic_directive_mutation_owner_authorised',
             'strategic_directive_mutation_blocked_non_owner_sender'
         )
         ORDER BY CAST(created_at AS INTEGER) DESC
         LIMIT ?1",
    )?;
    // We pull a generous slice and filter in Rust because the structured
    // thread/conversation match lives inside `details_json`. Clamp to a
    // sane upper bound so this stays cheap even if the audit trail is busy.
    let scan_limit = (limit.max(1) * 8).min(512) as i64;
    let rows = stmt.query_map(params![scan_limit], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, String>(3)?,
            row.get::<_, String>(4)?,
        ))
    })?;
    let mut out: Vec<StrategicDirectiveAuthorityEvent> = Vec::new();
    for row in rows {
        let (event_id, mechanism_id, severity, details_json, created_at) = row?;
        let details: serde_json::Value =
            serde_json::from_str(&details_json).unwrap_or(serde_json::Value::Null);
        let detail_thread = details
            .get("thread_key")
            .and_then(|value| value.as_str())
            .map(str::to_string);
        let detail_conversation = details
            .get("conversation_id")
            .and_then(|value| value.as_i64());
        let matches_thread = match thread_key {
            Some(key) => detail_thread.as_deref() == Some(key),
            None => false,
        };
        let matches_conversation = match (conversation_id, detail_conversation) {
            (Some(want), Some(got)) => want == got,
            _ => false,
        };
        if !matches_thread && !matches_conversation {
            continue;
        }
        out.push(StrategicDirectiveAuthorityEvent {
            event_id,
            mechanism_id,
            severity,
            created_at,
            sender_role: details
                .get("sender_role")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            sender_address: details
                .get("sender_address")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            directive_kind: details
                .get("directive_kind")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            attempted_status: details
                .get("attempted_status")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            action: details
                .get("action")
                .and_then(|value| value.as_str())
                .map(str::to_string),
            triggered_by_message_key: details
                .get("triggered_by_message_key")
                .and_then(|value| value.as_str())
                .map(str::to_string),
        });
        if out.len() >= limit {
            break;
        }
    }
    Ok(out)
}

/// JSON-friendly wrappers for the Business OS HTTP routes. These wrap the
/// existing internal channel functions without duplicating logic — the routes
/// in src/core/business_os/server.rs call these directly.

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
            let failure_reason = find_flag_value(args, "--reason");
            let message_keys = positional_after_flags(&args[1..]);
            if message_keys.is_empty() {
                anyhow::bail!(
                    "usage: ctox channel ack [--db <path>] [--status <status>] [--reason <text>] <message-key>..."
                );
            }
            let mut conn = open_channel_db(&db_path)?;
            let (failure_note, ack_reason) = if status == "failed" {
                (failure_reason, None)
            } else {
                (None, failure_reason)
            };
            let updated = ack_messages(&mut conn, &message_keys, status, failure_note, ack_reason)?;
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
        "founder-reply" => {
            anyhow::bail!(
                "direct founder-reply is disabled; founder/owner outbound email must be sent only through the reviewed service path"
            )
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
            let stored = ingest_tui_message(root, &mut conn, request)?;
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
        "context" => {
            let db_path = resolve_db_path(root, find_flag_value(args, "--db"));
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TAKE_LIMIT);
            let thread_key = required_flag_value(args, "--thread-key")?;
            let query = find_flag_value(args, "--query");
            let sender = find_flag_value(args, "--sender");
            let conn = open_channel_db(&db_path)?;
            let context = build_communication_context(&conn, thread_key, query, sender, limit)?;
            print_json(&json!({
                "ok": true,
                "db_path": db_path,
                "context": context,
            }))
        }
        "pipeline-status" => {
            let limit = find_flag_value(args, "--limit")
                .and_then(|value| value.parse::<usize>().ok())
                .unwrap_or(DEFAULT_TAKE_LIMIT);
            let thread_key = find_flag_value(args, "--thread-key");
            let report = pipeline_status(root, thread_key, limit)?;
            print_json(&json!({
                "ok": true,
                "report": report,
            }))
        }
        _ => {
            anyhow::bail!(
                "usage:\n  ctox channel init [--db <path>]\n  ctox channel sync --channel <email|jami|teams|meeting|whatsapp|slack|discord|telegram|matrix|mattermost|zulip|google_chat> [--db <path>] [adapter flags]\n  ctox channel take [--db <path>] [--channel <name>] [--limit <n>] [--lease-owner <owner>]\n  ctox channel ack [--db <path>] [--status <status>] <message-key>...\n  ctox channel send --channel <tui|email|jami|teams|meeting|whatsapp|slack|discord|telegram|matrix|mattermost|zulip|google_chat> --account-key <key> --thread-key <key> --body <text> [--subject <text>] [--to <addr>]... [--cc <addr>]... [--attach-file <path>]... [--send-voice] [--reviewed-founder-send] [--reviewed-communication-send]\n  ctox channel founder-reply --message-key <inbound-email-key> --body <text>\n  ctox channel test --channel <tui|email|jami|teams|whatsapp|slack|discord|telegram|matrix|mattermost|zulip|google_chat> [--db <path>] [--account-key <key>]\n  ctox channel ingest-tui --account-key <key> --thread-key <key> --body <text> [--sender-display <name>] [--sender-address <addr>] [--subject <text>]\n  ctox channel list [--db <path>] [--channel <name>] [--limit <n>]\n  ctox channel history --thread-key <key> [--db <path>] [--limit <n>]\n  ctox channel search --query <text> [--db <path>] [--channel <name>] [--sender <addr>] [--limit <n>]\n  ctox channel context --thread-key <key> [--db <path>] [--query <text>] [--sender <addr>] [--limit <n>]\n  ctox channel pipeline-status [--thread-key <key>] [--limit <n>]"
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
    refresh_inbound_priority_credits(root)?;
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
            let workspace_root =
                workspace_root_from_queue_metadata_or_prompt(&item.metadata, &item.body_text);
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
                workspace_root,
                metadata: item.metadata,
                preferred_reply_modality,
            }
        })
        .collect())
}

/// router-4: read-only, NON-leasing peek at the inbound messages the serial
/// router would lease this tick. Mirrors `take_messages` (no channel filter) — the
/// same eligibility `lease_pending_inbound_messages` uses (direction='inbound',
/// route_status pending, `not_before` elapsed, one row
/// per thread) — but performs NO lease UPDATE. Lets the router consult
/// `source_label_dispatch_rank` at the durable-queue-vs-inbound boundary without
/// consuming the message it is only inspecting.
pub fn peek_leasable_inbound_messages(
    root: &Path,
    limit: usize,
    lease_owner: &str,
) -> Result<Vec<RoutedInboundMessage>> {
    refresh_inbound_priority_credits(root)?;
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let mut statement = conn.prepare(
        r#"
        WITH eligible AS (
            SELECT
                m.message_key, m.channel, m.account_key, m.thread_key, m.remote_id,
                m.direction, m.folder_hint, m.sender_display, m.sender_address,
                m.subject, m.preview, m.body_text, m.status, m.seen,
                m.external_created_at, m.observed_at, m.metadata_json,
                r.route_status, r.lease_owner, r.leased_at, r.acked_at, r.last_error, r.updated_at,
                MIN(COALESCE(r.first_pending_at, m.external_created_at)) OVER (
                    PARTITION BY m.thread_key
                ) AS thread_pending_since,
                MIN(r.priority_time_credit_hours) OVER (
                    PARTITION BY m.thread_key
                ) AS thread_priority_credit_hours,
                ROW_NUMBER() OVER (
                    PARTITION BY m.thread_key
                    ORDER BY
                        CASE WHEN m.channel = 'queue' THEN m.external_created_at END ASC,
                        CASE WHEN m.channel <> 'queue' THEN m.external_created_at END DESC,
                        CASE WHEN m.channel = 'queue' THEN m.observed_at END ASC,
                        CASE WHEN m.channel <> 'queue' THEN m.observed_at END DESC,
                        m.message_key DESC
                ) AS thread_rank
            FROM communication_messages m
            JOIN communication_routing_state r ON r.message_key = m.message_key
            WHERE m.direction = 'inbound'
              AND r.route_status = 'pending'
              AND (r.retry_not_before IS NULL OR datetime(r.retry_not_before) <= datetime('now'))
              AND (
                    json_extract(m.metadata_json, '$.not_before') IS NULL
                 OR json_extract(m.metadata_json, '$.not_before') = ''
                 OR json_extract(m.metadata_json, '$.not_before') <= strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
              )
        )
        SELECT
            message_key, channel, account_key, thread_key, remote_id, direction,
            folder_hint, sender_display, sender_address, subject, preview, body_text,
            status, seen, external_created_at, observed_at, metadata_json,
            route_status, lease_owner, leased_at, acked_at, last_error, updated_at
        FROM eligible
        WHERE thread_rank = 1
        ORDER BY
            CASE
                WHEN channel = 'tui' THEN datetime(thread_pending_since, '-24 hours')
                WHEN channel = 'queue' THEN datetime(thread_pending_since, '+1 hour')
                ELSE datetime(thread_pending_since, printf('%+d hours', thread_priority_credit_hours))
            END ASC,
            CASE WHEN channel = 'queue' THEN external_created_at END ASC,
            CASE WHEN channel <> 'queue' THEN external_created_at END DESC,
            CASE WHEN channel = 'queue' THEN observed_at END ASC,
            CASE WHEN channel <> 'queue' THEN observed_at END DESC,
            message_key DESC
        LIMIT ?2
        "#,
    )?;
    let rows = statement.query_map(params![lease_owner, limit as i64], map_channel_message_row)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
        .map(|items| {
            items
                .into_iter()
                .map(routed_inbound_message_from_view)
                .collect()
        })
}

fn refresh_inbound_priority_credits(root: &Path) -> Result<()> {
    let settings = runtime_settings_with_owner_profiles(
        root,
        communication_gateway::CommunicationAdapterKind::Email,
    );
    let conn = open_channel_db(&resolve_db_path(root, None))?;
    let mut statement = conn.prepare(
        r#"
        SELECT m.message_key, m.sender_address
        FROM communication_messages m
        JOIN communication_routing_state r ON r.message_key=m.message_key
        WHERE m.direction='inbound' AND m.channel='email' AND r.route_status='pending'
        "#,
    )?;
    let rows = statement
        .query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    for (message_key, sender_address) in rows {
        let policy = classify_email_sender(&settings, &sender_address);
        let credit =
            if policy.allowed && matches!(policy.role.as_str(), "owner" | "founder" | "admin") {
                -24
            } else {
                0
            };
        conn.execute(
            "UPDATE communication_routing_state SET priority_time_credit_hours=?2 WHERE message_key=?1 AND priority_time_credit_hours!=?2",
            params![message_key, credit],
        )?;
    }
    Ok(())
}

pub fn list_stalled_inbound_messages(
    root: &Path,
    limit: usize,
) -> Result<Vec<RoutedInboundMessage>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
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
            r.last_error,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.direction = 'inbound'
          AND m.channel IN (
                'email', 'jami', 'teams', 'whatsapp', 'meeting', 'slack',
                'discord', 'telegram', 'matrix', 'mattermost', 'zulip',
                'google_chat'
          )
          AND r.route_status IN ('failed', 'review_rework')
          AND (
                r.acked_at IS NULL
             OR r.route_status IN ('failed', 'review_rework')
          )
        ORDER BY m.external_created_at DESC, m.observed_at DESC
        LIMIT ?1
        "#,
    )?;
    let rows = statement.query_map(params![limit as i64], map_channel_message_row)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
        .map(|items| {
            items
                .into_iter()
                .map(routed_inbound_message_from_view)
                .collect()
        })
}

pub fn list_unreviewed_handled_inbound_messages(
    root: &Path,
    limit: usize,
) -> Result<Vec<RoutedInboundMessage>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
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
            r.last_error,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.direction = 'inbound'
          AND m.channel IN ('email', 'jami')
          AND r.route_status = 'handled'
          AND NOT EXISTS (
              SELECT 1
              FROM communication_founder_reply_reviews review
              WHERE review.inbound_message_key = m.message_key
                AND review.sent_at IS NOT NULL
          )
        ORDER BY m.external_created_at DESC, m.observed_at DESC
        LIMIT ?1
        "#,
    )?;
    let rows = statement.query_map(params![limit as i64], map_channel_message_row)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
        .map(|items| {
            items
                .into_iter()
                .map(routed_inbound_message_from_view)
                .collect()
        })
}

pub fn founder_reply_sent_after_review_for_message(
    root: &Path,
    inbound_message_key: &str,
) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    founder_reply_sent_after_review(&conn, inbound_message_key)
}

/// Whether any inbound communication message is still pending or leased
/// (i.e. not acked as handled/blocked). Used by the mission watchdog to
/// avoid queuing redundant continuation tasks when real work is already
/// waiting in the channel queue.
pub fn has_runnable_inbound_message(root: &Path) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let count: i64 = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM communication_routing_state
        WHERE route_status IN ('pending', 'leased')
        "#,
        [],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

pub fn ack_leased_messages(root: &Path, message_keys: &[String], status: &str) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    guard_founder_handled_ack(root, &conn, message_keys, status)?;
    ack_messages(&mut conn, message_keys, status, None, None)
}

/// Ack with an explicit routing reason. The reason feeds the core-transition
/// audit trail and, for terminal-success policy paths (e.g. an inbound
/// message fully handled by scheduling a meeting), selects the matching
/// `queue_terminal_policy_proof` entry.
pub fn ack_leased_messages_with_reason(
    root: &Path,
    message_keys: &[String],
    status: &str,
    reason: &str,
) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    guard_founder_handled_ack(root, &conn, message_keys, status)?;
    ack_messages(&mut conn, message_keys, status, None, Some(reason))
}

pub fn ack_leased_messages_with_failure_reason(
    root: &Path,
    message_keys: &[String],
    status: &str,
    failure_reason: &str,
) -> Result<usize> {
    anyhow::ensure!(
        status == "failed",
        "ack_leased_messages_with_failure_reason only accepts status='failed'"
    );
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    guard_founder_handled_ack(root, &conn, message_keys, status)?;
    ack_messages(&mut conn, message_keys, status, Some(failure_reason), None)
}

/// Persist a typed completion hold without allowing an unbounded
/// pending→leased→pending loop. External waits become dormant `blocked` rows;
/// technical/evidence/artifact holds consume the existing five-attempt review
/// budget with exponential backoff and terminalize when exhausted.
pub fn hold_leased_messages(
    root: &Path,
    message_keys: &[String],
    reason: &HoldReason,
    summary: &str,
) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let tx = conn.transaction()?;
    let now = now_iso_string();
    let mut updated = 0usize;
    for message_key in message_keys {
        match reason {
            HoldReason::WaitingExternal(wait_ref) => {
                let command_transitioned = transition_business_command_for_task_in_transaction(
                    &tx,
                    message_key,
                    "blocked",
                    None,
                    None,
                    Some(summary.trim()),
                    "waiting_external",
                )?;
                if !command_transitioned {
                    updated += ack_messages_in_transaction(
                        &tx,
                        std::slice::from_ref(message_key),
                        "blocked",
                        None,
                        Some("waiting_external"),
                    )?;
                } else {
                    updated += 1;
                }
                anyhow::ensure!(
                    current_queue_route_status(&tx, message_key)? == "blocked",
                    "waiting-external hold did not persist the linked queue route"
                );
                tx.execute(
                    r#"
                    UPDATE communication_routing_state
                    SET hold_reason='waiting_external', wait_entity_type=?2,
                        wait_entity_id=?3, retry_not_before=NULL,
                        lease_expires_at=NULL, lease_worker_id=NULL, last_error=?4, updated_at=?5
                    WHERE message_key=?1
                    "#,
                    params![
                        message_key,
                        wait_ref.entity_type,
                        wait_ref.entity_id,
                        summary.trim(),
                        now,
                    ],
                )?;
            }
            HoldReason::Technical { .. }
            | HoldReason::MissingReviewEvidence
            | HoldReason::MissingArtifact => {
                let previous_attempts: i64 = tx
                    .query_row(
                        "SELECT failure_attempt_count FROM communication_routing_state WHERE message_key=?1",
                        params![message_key],
                        |row| row.get(0),
                    )
                    .optional()?
                    .unwrap_or(0);
                let attempts = previous_attempts.saturating_add(1);
                let exhausted = attempts >= 5;
                let failure_class = match reason {
                    HoldReason::Technical { .. } => "technical",
                    HoldReason::MissingReviewEvidence => "missing_review_evidence",
                    HoldReason::MissingArtifact => "missing_artifact",
                    HoldReason::WaitingExternal(_) => unreachable!(),
                };
                let hold_reason = match reason {
                    HoldReason::Technical { policy_id } => format!("technical:{policy_id}"),
                    HoldReason::MissingReviewEvidence => "missing_review_evidence".to_string(),
                    HoldReason::MissingArtifact => "missing_artifact".to_string(),
                    HoldReason::WaitingExternal(_) => unreachable!(),
                };
                let next_status = if exhausted { "failed" } else { "pending" };
                let retry_not_before = (!exhausted).then(|| {
                    let exponent = u32::try_from(attempts.saturating_sub(1))
                        .unwrap_or(16)
                        .min(16);
                    let seconds = 300_i64
                        .saturating_mul(2_i64.saturating_pow(exponent))
                        .min(3_600);
                    (Utc::now() + Duration::seconds(seconds)).to_rfc3339()
                });
                let command_transitioned = transition_business_command_for_task_in_transaction(
                    &tx,
                    message_key,
                    next_status,
                    None,
                    None,
                    Some(summary.trim()),
                    "budgeted_completion_hold",
                )?;
                if !command_transitioned {
                    updated += ack_messages_in_transaction(
                        &tx,
                        std::slice::from_ref(message_key),
                        next_status,
                        exhausted.then_some(summary.trim()),
                        (!exhausted).then_some("budgeted_completion_hold"),
                    )?;
                } else {
                    updated += 1;
                }
                anyhow::ensure!(
                    current_queue_route_status(&tx, message_key)? == next_status,
                    "budgeted completion hold did not persist the linked queue route"
                );
                tx.execute(
                    r#"
                    UPDATE communication_routing_state
                    SET failure_class=?2, failure_attempt_count=?3,
                        retry_not_before=?4, hold_reason=?5,
                        wait_entity_type=NULL, wait_entity_id=NULL,
                        lease_expires_at=NULL, lease_worker_id=NULL, last_error=?6, updated_at=?7
                    WHERE message_key=?1
                    "#,
                    params![
                        message_key,
                        failure_class,
                        attempts,
                        retry_not_before,
                        hold_reason,
                        summary.trim(),
                        now,
                    ],
                )?;
            }
        }
    }
    tx.commit()?;
    Ok(updated)
}

pub fn wake_messages_waiting_for(root: &Path, entity_type: &str, entity_id: &str) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let tx = conn.unchecked_transaction()?;
    let now = now_iso_string();
    let message_keys = {
        let mut statement = tx.prepare(
            r#"
            SELECT message_key
            FROM communication_routing_state
            WHERE route_status='blocked'
              AND hold_reason='waiting_external'
              AND LOWER(wait_entity_type)=LOWER(?1)
              AND wait_entity_id=?2
            "#,
        )?;
        let rows = statement
            .query_map(params![entity_type.trim(), entity_id.trim()], |row| {
                row.get::<_, String>(0)
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
    };
    let mut updated = 0usize;
    for message_key in message_keys {
        let changed = tx.execute(
            r#"UPDATE communication_routing_state
               SET route_status='pending', hold_reason=NULL, wait_entity_type=NULL,
                   wait_entity_id=NULL, retry_not_before=NULL, first_pending_at=?2, updated_at=?2
               WHERE message_key=?1 AND route_status='blocked' AND hold_reason='waiting_external'"#,
            params![message_key, now],
        )?;
        if changed != 0 {
            enforce_queue_route_status_transition(
                &tx,
                &message_key,
                "blocked",
                "pending",
                "ctox-wait-wakeup",
                "wake_messages_waiting_for",
            )?;
            updated += changed;
        }
    }
    tx.commit()?;
    Ok(updated)
}

pub fn set_queue_task_route_status(
    root: &Path,
    message_key: &str,
    route_status: &str,
) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    if load_queue_message_from_conn(&conn, message_key)?.is_none() {
        return Ok(false);
    }
    update_queue_task(
        root,
        QueueTaskUpdateRequest {
            message_key: message_key.to_string(),
            route_status: Some(route_status.to_string()),
            ..Default::default()
        },
    )?;
    Ok(true)
}

pub fn create_queue_task(root: &Path, request: QueueTaskCreateRequest) -> Result<QueueTaskView> {
    create_queue_task_with_metadata(root, request)
}

pub fn create_queue_task_with_metadata(
    root: &Path,
    request: QueueTaskCreateRequest,
) -> Result<QueueTaskView> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let tx = conn.transaction()?;
    let task = create_queue_task_with_metadata_tx(&tx, request)?;
    tx.commit()?;
    Ok(task)
}

fn create_queue_task_with_metadata_tx(
    tx: &Transaction<'_>,
    request: QueueTaskCreateRequest,
) -> Result<QueueTaskView> {
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
    // ref: tickets.rs:4603-4617 — honor a caller-supplied idempotency key so a
    // crash-retried create folds to the same message_key (stable edge_id +
    // ON CONFLICT(message_key) DO UPDATE); default to now-salt when absent so
    // distinct callers are unaffected.
    let idempotency_key = request
        .extra_metadata
        .as_ref()
        .and_then(|extra| metadata_string_value(extra, "idempotency_key"));
    let digest = stable_digest(&format!(
        "{}:{}:{}:{}",
        title,
        prompt,
        request.thread_key.trim(),
        idempotency_key.as_deref().unwrap_or(now.as_str())
    ));
    let message_key = format!("{QUEUE_ACCOUNT_KEY}::{digest}");
    let remote_id = format!("queue-{digest}");
    let mut metadata = json!({
        "source": "ctox-queue",
        "priority": priority,
        "skill": request.suggested_skill.as_deref(),
        "parent_message_key": request.parent_message_key.as_deref(),
        "workspace_root": normalize_workspace_root(request.workspace_root.as_deref())
            .or_else(|| legacy_workspace_root_from_prompt(prompt)),
        "created_at": now,
        "sort_at": sort_at,
    });
    if let Some(extra) = request.extra_metadata {
        merge_object_metadata(&mut metadata, extra);
    }
    enforce_queue_task_spawn(
        tx,
        &metadata,
        request.parent_message_key.as_deref(),
        request.thread_key.trim(),
        &message_key,
        title,
    )?;
    upsert_communication_message_tx(
        tx,
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
    refresh_thread_tx(tx, request.thread_key.trim())?;
    ensure_routing_rows_for_inbound(tx)?;
    load_queue_task_from_conn(tx, &message_key)?.context("failed to load created queue task")
}

pub(crate) fn claim_business_command_with_queue(
    root: &Path,
    claim: BusinessCommandClaimRequest,
    request: QueueTaskCreateRequest,
) -> Result<BusinessCommandQueueClaim> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let tx = conn.transaction()?;
    let existing = tx
        .query_row(
            "SELECT idempotency_key, payload_hash, execution_phase, projection_version
             FROM business_command_aggregates
             WHERE command_id = ?1",
            params![claim.command_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, i64>(3)?,
                ))
            },
        )
        .optional()?;
    let (aggregate_exists, from_phase, accepted_version) =
        if let Some((idempotency_key, payload_hash, phase, version)) = existing {
            anyhow::ensure!(
                idempotency_key == claim.idempotency_key && payload_hash == claim.payload_hash,
                "idempotency_conflict: command id was already claimed with different intent"
            );
            let task_id = tx
                .query_row(
                    "SELECT task_id FROM business_command_task_links WHERE command_id = ?1",
                    params![claim.command_id],
                    |row| row.get::<_, String>(0),
                )
                .optional()?;
            if let Some(task_id) = task_id {
                let task = load_queue_task_from_conn(&tx, &task_id)?
                    .context("claimed queue command task link points to a missing task")?;
                tx.commit()?;
                return Ok(BusinessCommandQueueClaim {
                    task,
                    already_claimed: true,
                });
            }
            anyhow::ensure!(
                phase == "waiting_dependencies",
                "claimed queue command is missing its atomic task link"
            );
            (true, phase, version.saturating_add(1))
        } else {
            (false, "local".to_string(), 1)
        };

    let now_ms = epoch_millis();
    if from_phase != "local" {
        crate::business_os::command_lifecycle::validate_execution_phase_transition(
            &from_phase,
            "accepted",
        )?;
    }
    crate::business_os::command_lifecycle::validate_execution_phase_transition(
        "accepted", "queued",
    )?;
    if !aggregate_exists {
        tx.execute(
            "INSERT INTO business_command_aggregates
            (command_id, idempotency_key, payload_hash, module, command_type, record_id,
             execution_mode, execution_phase, terminal_status, attempt, projection_version,
             intent_json, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'queue', 'accepted', 'none', 0, 1, ?7, ?8, ?9)",
            params![
                claim.command_id,
                claim.idempotency_key,
                claim.payload_hash,
                claim.module,
                claim.command_type,
                claim.record_id,
                serde_json::to_string(&claim.intent)?,
                claim.created_at_ms,
                now_ms,
            ],
        )?;
    } else {
        tx.execute(
            "UPDATE business_command_aggregates
             SET execution_phase = 'accepted', projection_version = ?2, updated_at_ms = ?3
             WHERE command_id = ?1",
            params![claim.command_id, accepted_version, now_ms],
        )?;
    }
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, ?2, ?3, 'accepted', 'none', 'canonical command admission', '{}', ?4)",
        params![claim.command_id, accepted_version, from_phase, now_ms],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        &claim.command_id,
        accepted_version,
        "command.accepted",
        &json!({
            "command_id": claim.command_id,
            "execution_mode": "queue",
            "execution_phase": "accepted",
            "terminal_status": "none",
            "projection_version": accepted_version,
        }),
        now_ms,
    )?;
    let task = create_queue_task_with_metadata_tx(&tx, request)?;
    let queued_version = accepted_version.saturating_add(1);
    tx.execute(
        "INSERT INTO business_command_task_links (command_id, task_id, created_at_ms)
         VALUES (?1, ?2, ?3)",
        params![claim.command_id, task.message_key, now_ms],
    )?;
    tx.execute(
        "INSERT INTO business_command_effects
            (command_id, effect_key, status, claimed_at_ms, updated_at_ms)
         VALUES (?1, ?2, 'claimed', ?3, ?3)",
        params![
            claim.command_id,
            format!("queue:{}", task.message_key),
            now_ms
        ],
    )?;
    tx.execute(
        "UPDATE business_command_aggregates
         SET execution_phase = 'queued', projection_version = ?2, updated_at_ms = ?3
         WHERE command_id = ?1",
        params![claim.command_id, queued_version, now_ms],
    )?;
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, ?2, ?3, 'queued', 'none', 'atomic queue admission', ?4, ?5)",
        params![
            claim.command_id,
            queued_version,
            "accepted",
            serde_json::to_string(&json!({ "task_id": task.message_key }))?,
            now_ms,
        ],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        &claim.command_id,
        queued_version,
        "command.admitted",
        &json!({
            "command_id": claim.command_id,
            "execution_mode": "queue",
            "execution_task_id": task.message_key,
            "execution_phase": "queued",
            "terminal_status": "none",
            "projection_version": queued_version,
        }),
        now_ms,
    )?;
    tx.commit()?;
    Ok(BusinessCommandQueueClaim {
        task,
        already_claimed: false,
    })
}

const MODULE_VISIBILITY_SAGA_STEPS: &[(&str, &str, &str)] = &[
    (
        "persist_visibility",
        "module_visibility:persist",
        "module_visibility:restore",
    ),
    (
        "project_catalog",
        "module_visibility:project",
        "module_visibility:reproject",
    ),
];

fn registered_business_command_saga(
    command_type: &str,
) -> Option<(
    &'static str,
    &'static [(&'static str, &'static str, &'static str)],
)> {
    match command_type {
        "ctox.module.set_visible" => {
            Some(("ctox.module.visibility.v1", MODULE_VISIBILITY_SAGA_STEPS))
        }
        _ => None,
    }
}

fn register_business_command_saga_tx(
    tx: &Transaction<'_>,
    command_id: &str,
    command_type: &str,
    now_ms: i64,
) -> Result<()> {
    let Some((saga_kind, steps)) = registered_business_command_saga(command_type) else {
        return Ok(());
    };
    let saga_id = format!("saga:{command_id}");
    tx.execute(
        "INSERT OR IGNORE INTO business_command_sagas
            (saga_id, command_id, saga_kind, phase, current_step, total_steps, compensation_status, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, ?3, 'forward', 0, ?4, 'not_started', ?5, ?5)",
        params![saga_id, command_id, saga_kind, steps.len() as i64, now_ms],
    )?;
    for (index, (name, forward_key, compensation_key)) in steps.iter().enumerate() {
        tx.execute(
            "INSERT OR IGNORE INTO business_command_saga_steps
                (saga_id, step_index, step_name, forward_effect_key, compensation_effect_key, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![saga_id, index as i64, name, forward_key, compensation_key, now_ms],
        )?;
    }
    Ok(())
}

/// Register the immutable definition and ordered effects for a runtime-loaded
/// app action. Runtime actions intentionally share the same durable saga
/// tables and terminal owner as compiled commands; only their definition is
/// loaded from the validated module package.
pub(crate) fn start_runtime_business_command_saga(
    root: &Path,
    command_id: &str,
    module_id: &str,
    action_name: &str,
    definition_hash: &str,
    definition: &Value,
    step_names: &[String],
) -> Result<()> {
    anyhow::ensure!(
        !step_names.is_empty(),
        "app action saga requires at least one step"
    );
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let now_ms = epoch_millis();
    let saga_id = format!("saga:{command_id}");
    tx.execute(
        "INSERT OR IGNORE INTO business_app_action_snapshots
            (command_id, module_id, action_name, definition_hash, definition_json, created_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
        params![
            command_id,
            module_id,
            action_name,
            definition_hash,
            serde_json::to_string(definition)?,
            now_ms,
        ],
    )?;
    let existing: (String, String) = tx.query_row(
        "SELECT definition_hash, definition_json FROM business_app_action_snapshots WHERE command_id = ?1",
        params![command_id],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )?;
    anyhow::ensure!(
        existing.0 == definition_hash && existing.1 == serde_json::to_string(definition)?,
        "app_action_definition_changed: command already admitted with another definition"
    );
    tx.execute(
        "INSERT OR IGNORE INTO business_command_sagas
            (saga_id, command_id, saga_kind, phase, current_step, total_steps, compensation_status, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, ?3, 'forward', 0, ?4, 'not_started', ?5, ?5)",
        params![
            saga_id,
            command_id,
            format!("ctox.app.action.v1:{module_id}:{action_name}:{definition_hash}"),
            step_names.len() as i64,
            now_ms,
        ],
    )?;
    for (index, name) in step_names.iter().enumerate() {
        let forward_key = format!("{command_id}:{definition_hash}:{index}:forward");
        let compensation_key = format!("{command_id}:{definition_hash}:{index}:compensation");
        tx.execute(
            "INSERT OR IGNORE INTO business_command_saga_steps
                (saga_id, step_index, step_name, forward_effect_key, compensation_effect_key, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![saga_id, index as i64, name, forward_key, compensation_key, now_ms],
        )?;
    }
    let registered_steps: i64 = tx.query_row(
        "SELECT COUNT(*) FROM business_command_saga_steps WHERE saga_id = ?1",
        params![saga_id],
        |row| row.get(0),
    )?;
    anyhow::ensure!(
        registered_steps == step_names.len() as i64,
        "app_action_definition_changed: registered saga step count differs"
    );
    tx.commit()?;
    Ok(())
}

pub(crate) fn runtime_business_command_action_snapshot(
    root: &Path,
    command_id: &str,
) -> Result<Option<Value>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let raw: Option<String> = conn
        .query_row(
            "SELECT definition_json FROM business_app_action_snapshots WHERE command_id = ?1",
            params![command_id],
            |row| row.get(0),
        )
        .optional()?;
    raw.map(|value| serde_json::from_str(&value).map_err(Into::into))
        .transpose()
}

pub(crate) fn business_command_saga_status(root: &Path, command_id: &str) -> Result<Option<Value>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let saga_id = format!("saga:{command_id}");
    let saga: Option<(String, String, i64, i64)> = conn
        .query_row(
            "SELECT phase, compensation_status, current_step, total_steps
             FROM business_command_sagas WHERE saga_id = ?1",
            params![saga_id],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .optional()?;
    let Some((phase, compensation_status, current_step, total_steps)) = saga else {
        return Ok(None);
    };
    let error_message: Option<String> = conn
        .query_row(
            "SELECT error_message FROM business_command_saga_steps
             WHERE saga_id = ?1 AND error_message IS NOT NULL
             ORDER BY CASE WHEN compensation_status = 'failed' THEN 0 ELSE 1 END, step_index DESC
             LIMIT 1",
            params![format!("saga:{command_id}")],
            |row| row.get(0),
        )
        .optional()?;
    Ok(Some(json!({
        "phase": phase,
        "compensation_status": compensation_status,
        "current_step": current_step,
        "total_steps": total_steps,
        "error_message": error_message,
    })))
}

pub(crate) fn start_business_command_saga(
    root: &Path,
    command_id: &str,
    command_type: &str,
) -> Result<()> {
    anyhow::ensure!(
        registered_business_command_saga(command_type).is_some(),
        "no native saga definition registered for `{command_type}`"
    );
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    register_business_command_saga_tx(&tx, command_id, command_type, epoch_millis())?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn claim_business_command_saga_step(
    root: &Path,
    command_id: &str,
    step_name: &str,
    compensation: bool,
) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let saga_id = format!("saga:{command_id}");
    let column = if compensation {
        "compensation_status"
    } else {
        "forward_status"
    };
    let attempts = if compensation {
        "compensation_attempts"
    } else {
        "forward_attempts"
    };
    let step: Option<(String, i64, String)> = tx.query_row(
        &format!("SELECT s.{column}, s.step_index, g.phase FROM business_command_saga_steps s JOIN business_command_sagas g ON g.saga_id = s.saga_id WHERE s.saga_id = ?1 AND s.step_name = ?2"),
        params![saga_id, step_name],
        |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
    ).optional()?;
    let (status, step_index, saga_phase) =
        step.with_context(|| format!("no registered saga step `{step_name}` for `{command_id}`"))?;
    if status == "completed" {
        tx.commit()?;
        return Ok(false);
    }
    if compensation {
        anyhow::ensure!(
            saga_phase == "compensating" || saga_phase == "manual_intervention",
            "saga is not compensating"
        );
        anyhow::ensure!(
            status != "not_required",
            "compensation was not requested for saga step `{step_name}`"
        );
        let later_pending: i64 = tx.query_row(
            "SELECT COUNT(*) FROM business_command_saga_steps
             WHERE saga_id = ?1 AND step_index > ?2 AND compensation_status IN ('pending', 'claimed', 'failed')",
            params![saga_id, step_index],
            |row| row.get(0),
        )?;
        anyhow::ensure!(
            later_pending == 0,
            "saga compensation must run in reverse step order"
        );
    } else {
        anyhow::ensure!(saga_phase == "forward", "saga is not in forward phase");
        let earlier_incomplete: i64 = tx.query_row(
            "SELECT COUNT(*) FROM business_command_saga_steps
             WHERE saga_id = ?1 AND step_index < ?2 AND forward_status != 'completed'",
            params![saga_id, step_index],
            |row| row.get(0),
        )?;
        anyhow::ensure!(
            earlier_incomplete == 0,
            "saga forward steps must run in registered order"
        );
    }
    let now_ms = epoch_millis();
    tx.execute(
        &format!("UPDATE business_command_saga_steps SET {column} = 'claimed', {attempts} = {attempts} + 1, updated_at_ms = ?3 WHERE saga_id = ?1 AND step_name = ?2"),
        params![saga_id, step_name, now_ms],
    )?;
    tx.execute(
        "UPDATE business_command_sagas SET current_step = (SELECT step_index FROM business_command_saga_steps WHERE saga_id = ?1 AND step_name = ?2), updated_at_ms = ?3 WHERE saga_id = ?1",
        params![saga_id, step_name, now_ms],
    )?;
    tx.commit()?;
    Ok(true)
}

pub(crate) fn business_command_saga_step_evidence(
    root: &Path,
    command_id: &str,
    step_name: &str,
) -> Result<Value> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let raw: String = conn.query_row(
        "SELECT evidence_json FROM business_command_saga_steps WHERE saga_id = ?1 AND step_name = ?2",
        params![format!("saga:{command_id}"), step_name],
        |row| row.get(0),
    )?;
    Ok(serde_json::from_str(&raw)?)
}

pub(crate) fn business_command_saga_pending_compensation_steps(
    root: &Path,
    command_id: &str,
) -> Result<Vec<String>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let saga_id = format!("saga:{command_id}");
    let mut stmt = conn.prepare(
        "SELECT step_name FROM business_command_saga_steps
         WHERE saga_id = ?1 AND compensation_status IN ('pending', 'claimed')
         ORDER BY step_index ASC",
    )?;
    let rows = stmt.query_map(params![saga_id], |row| row.get::<_, String>(0))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

pub(crate) fn record_business_command_saga_step_evidence(
    root: &Path,
    command_id: &str,
    step_name: &str,
    evidence: &Value,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let changed = conn.execute(
        "UPDATE business_command_saga_steps SET evidence_json = ?3, updated_at_ms = ?4
         WHERE saga_id = ?1 AND step_name = ?2 AND forward_status = 'claimed'",
        params![
            format!("saga:{command_id}"),
            step_name,
            serde_json::to_string(evidence)?,
            epoch_millis(),
        ],
    )?;
    anyhow::ensure!(changed == 1, "saga step `{step_name}` is not claimed");
    Ok(())
}

pub(crate) fn complete_business_command_saga_step(
    root: &Path,
    command_id: &str,
    step_name: &str,
    compensation: bool,
    evidence: &Value,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let saga_id = format!("saga:{command_id}");
    let column = if compensation {
        "compensation_status"
    } else {
        "forward_status"
    };
    let now_ms = epoch_millis();
    let changed = tx.execute(
        &format!("UPDATE business_command_saga_steps SET {column} = 'completed', evidence_json = ?3, error_message = NULL, updated_at_ms = ?4 WHERE saga_id = ?1 AND step_name = ?2 AND {column} IN ('claimed', 'completed')"),
        params![saga_id, step_name, serde_json::to_string(evidence)?, now_ms],
    )?;
    anyhow::ensure!(
        changed == 1,
        "saga step `{step_name}` was not durably claimed"
    );
    if compensation {
        let pending: i64 = tx.query_row(
            "SELECT COUNT(*) FROM business_command_saga_steps WHERE saga_id = ?1 AND compensation_status IN ('pending', 'claimed', 'failed')",
            params![saga_id], |row| row.get(0),
        )?;
        if pending == 0 {
            tx.execute("UPDATE business_command_sagas SET phase = 'compensated', compensation_status = 'completed', updated_at_ms = ?2 WHERE saga_id = ?1", params![saga_id, now_ms])?;
        }
    } else {
        let pending: i64 = tx.query_row(
            "SELECT COUNT(*) FROM business_command_saga_steps WHERE saga_id = ?1 AND forward_status != 'completed'",
            params![saga_id], |row| row.get(0),
        )?;
        if pending == 0 {
            tx.execute("UPDATE business_command_sagas SET phase = 'completed', current_step = total_steps, updated_at_ms = ?2 WHERE saga_id = ?1", params![saga_id, now_ms])?;
        }
    }
    tx.commit()?;
    Ok(())
}

pub(crate) fn fail_business_command_saga_step(
    root: &Path,
    command_id: &str,
    step_name: &str,
    error_message: &str,
    compensation: bool,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let saga_id = format!("saga:{command_id}");
    let column = if compensation {
        "compensation_status"
    } else {
        "forward_status"
    };
    let now_ms = epoch_millis();
    tx.execute(
        &format!("UPDATE business_command_saga_steps SET {column} = 'failed', error_message = ?3, updated_at_ms = ?4 WHERE saga_id = ?1 AND step_name = ?2"),
        params![saga_id, step_name, error_message, now_ms],
    )?;
    if compensation {
        tx.execute("UPDATE business_command_sagas SET phase = 'manual_intervention', compensation_status = 'failed', updated_at_ms = ?2 WHERE saga_id = ?1", params![saga_id, now_ms])?;
    } else {
        tx.execute(
            "UPDATE business_command_saga_steps SET compensation_status = 'pending', updated_at_ms = ?2
             WHERE saga_id = ?1 AND forward_status = 'completed'",
            params![saga_id, now_ms],
        )?;
        let compensation_count: i64 = tx.query_row(
            "SELECT COUNT(*) FROM business_command_saga_steps WHERE saga_id = ?1 AND compensation_status = 'pending'",
            params![saga_id], |row| row.get(0),
        )?;
        tx.execute(
            "UPDATE business_command_sagas SET phase = ?2, compensation_status = ?3, updated_at_ms = ?4 WHERE saga_id = ?1",
            params![
                saga_id,
                if compensation_count == 0 { "compensated" } else { "compensating" },
                if compensation_count == 0 { "completed" } else { "pending" },
                now_ms,
            ],
        )?;
    }
    tx.commit()?;
    Ok(())
}

pub(crate) fn claim_business_command_waiting_dependencies(
    root: &Path,
    claim: BusinessCommandClaimRequest,
    missing_dependencies: &Value,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let existing = tx
        .query_row(
            "SELECT idempotency_key, payload_hash FROM business_command_aggregates WHERE command_id = ?1",
            params![claim.command_id],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?;
    if let Some((idempotency_key, payload_hash)) = existing {
        anyhow::ensure!(
            idempotency_key == claim.idempotency_key && payload_hash == claim.payload_hash,
            "idempotency_conflict: command id was already claimed with different intent"
        );
        tx.commit()?;
        return Ok(());
    }
    let now_ms = epoch_millis();
    tx.execute(
        "INSERT INTO business_command_aggregates
            (command_id, idempotency_key, payload_hash, module, command_type, record_id,
             execution_mode, execution_phase, terminal_status, attempt, projection_version,
             intent_json, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'queue', 'waiting_dependencies', 'none', 0, 1, ?7, ?8, ?9)",
        params![
            claim.command_id,
            claim.idempotency_key,
            claim.payload_hash,
            claim.module,
            claim.command_type,
            claim.record_id,
            serde_json::to_string(&claim.intent)?,
            claim.created_at_ms,
            now_ms,
        ],
    )?;
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, 1, 'local', 'waiting_dependencies', 'none', 'required replicated data is unavailable', ?2, ?3)",
        params![claim.command_id, serde_json::to_string(missing_dependencies)?, now_ms],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        &claim.command_id,
        1,
        "command.waiting_dependencies",
        &json!({
            "command_id": claim.command_id,
            "execution_mode": "queue",
            "execution_phase": "waiting_dependencies",
            "terminal_status": "none",
            "projection_version": 1,
            "missing_dependencies": missing_dependencies,
        }),
        now_ms,
    )?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn claim_business_control_command(
    root: &Path,
    claim: BusinessCommandClaimRequest,
) -> Result<BusinessCommandControlClaim> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let existing = tx
        .query_row(
            "SELECT idempotency_key, payload_hash, terminal_status, result_json, execution_phase
             FROM business_command_aggregates
             WHERE command_id = ?1",
            params![claim.command_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()?;
    if let Some((idempotency_key, payload_hash, terminal_status, result_json, phase)) = existing {
        anyhow::ensure!(
            idempotency_key == claim.idempotency_key && payload_hash == claim.payload_hash,
            "idempotency_conflict: command id was already claimed with different intent"
        );
        let result = result_json
            .as_deref()
            .map(serde_json::from_str)
            .transpose()?;
        tx.commit()?;
        return Ok(BusinessCommandControlClaim {
            disposition: if phase == "terminal" {
                "terminal"
            } else {
                "uncertain"
            },
            result,
            terminal_status: (terminal_status != "none").then_some(terminal_status),
        });
    }

    let now_ms = epoch_millis();
    tx.execute(
        "INSERT INTO business_command_aggregates
            (command_id, idempotency_key, payload_hash, module, command_type, record_id,
             execution_mode, execution_phase, terminal_status, attempt, projection_version,
             intent_json, created_at_ms, updated_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'control', 'accepted', 'none', 0, 1, ?7, ?8, ?9)",
        params![
            claim.command_id,
            claim.idempotency_key,
            claim.payload_hash,
            claim.module,
            claim.command_type,
            claim.record_id,
            serde_json::to_string(&claim.intent)?,
            claim.created_at_ms,
            now_ms,
        ],
    )?;
    tx.execute(
        "INSERT INTO business_command_effects
            (command_id, effect_key, status, claimed_at_ms, updated_at_ms)
         VALUES (?1, ?2, 'claimed', ?3, ?3)",
        params![
            claim.command_id,
            format!("control:{}", claim.command_type),
            now_ms,
        ],
    )?;
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, 1, 'local', 'accepted', 'none', 'durable control claim', '{}', ?2)",
        params![claim.command_id, now_ms],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        &claim.command_id,
        1,
        "command.claimed",
        &json!({
            "command_id": claim.command_id,
            "execution_mode": "control",
            "execution_phase": "accepted",
            "terminal_status": "none",
            "projection_version": 1,
        }),
        now_ms,
    )?;
    tx.commit()?;
    Ok(BusinessCommandControlClaim {
        disposition: "new",
        result: None,
        terminal_status: None,
    })
}

pub(crate) fn complete_business_control_command(
    root: &Path,
    command_id: &str,
    terminal_status: &str,
    result: &Value,
    error_message: Option<&str>,
) -> Result<()> {
    anyhow::ensure!(
        matches!(terminal_status, "completed" | "failed" | "cancelled"),
        "invalid control command terminal status"
    );
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let (phase, version, command_type) = tx.query_row(
        "SELECT execution_phase, projection_version, command_type
         FROM business_command_aggregates WHERE command_id = ?1",
        params![command_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ))
        },
    )?;
    if phase == "terminal" {
        tx.commit()?;
        return Ok(());
    }
    let saga_phase: Option<String> = tx
        .query_row(
            "SELECT phase FROM business_command_sagas WHERE command_id = ?1",
            params![command_id],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(saga_phase) = saga_phase.as_deref() {
        if terminal_status == "completed" {
            anyhow::ensure!(
                saga_phase == "completed",
                "terminal success rejected: command saga is `{saga_phase}`"
            );
        } else {
            anyhow::ensure!(
                matches!(saga_phase, "compensated" | "manual_intervention"),
                "terminal failure rejected until saga compensation is durably captured (phase `{saga_phase}`)"
            );
        }
    }
    crate::business_os::command_lifecycle::validate_execution_phase_transition(&phase, "terminal")?;
    let next_version = version.saturating_add(1);
    let now_ms = epoch_millis();
    let error_code = result.get("error_code").and_then(Value::as_str);
    tx.execute(
        "UPDATE business_command_aggregates
         SET execution_phase = 'terminal', terminal_status = ?2,
             projection_version = ?3, result_json = ?4, error_code = ?5,
             error_message = ?6, retryable = 0, updated_at_ms = ?7
         WHERE command_id = ?1",
        params![
            command_id,
            terminal_status,
            next_version,
            serde_json::to_string(result)?,
            error_code,
            error_message,
            now_ms,
        ],
    )?;
    tx.execute(
        "UPDATE business_command_effects
         SET status = ?3, result_json = ?4, error_message = ?5, updated_at_ms = ?6
         WHERE command_id = ?1 AND effect_key = ?2",
        params![
            command_id,
            format!("control:{command_type}"),
            if terminal_status == "completed" {
                "completed"
            } else {
                "failed"
            },
            serde_json::to_string(result)?,
            error_message,
            now_ms,
        ],
    )?;
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, ?2, ?3, 'terminal', ?4, 'control effect outcome persisted', ?5, ?6)",
        params![
            command_id,
            next_version,
            phase,
            terminal_status,
            serde_json::to_string(result)?,
            now_ms,
        ],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        command_id,
        next_version,
        "command.terminal",
        &json!({
            "command_id": command_id,
            "execution_mode": "control",
            "execution_phase": "terminal",
            "terminal_status": terminal_status,
            "projection_version": next_version,
            "result": result,
        }),
        now_ms,
    )?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn progress_business_control_command(
    root: &Path,
    command_id: &str,
    execution_phase: &str,
    result: &Value,
) -> Result<()> {
    anyhow::ensure!(
        matches!(execution_phase, "running"),
        "invalid control command progress phase"
    );
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let (phase, version) = tx.query_row(
        "SELECT execution_phase, projection_version
         FROM business_command_aggregates WHERE command_id = ?1",
        params![command_id],
        |row| Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?)),
    )?;
    if phase == "terminal" || phase == execution_phase {
        tx.commit()?;
        return Ok(());
    }
    crate::business_os::command_lifecycle::validate_execution_phase_transition(
        &phase,
        execution_phase,
    )?;
    let next_version = version.saturating_add(1);
    let now_ms = epoch_millis();
    tx.execute(
        "UPDATE business_command_aggregates
         SET execution_phase = ?2, terminal_status = 'none', projection_version = ?3,
             result_json = ?4, retryable = 0,
             attempt = attempt + CASE WHEN execution_phase != ?2 THEN 1 ELSE 0 END,
             updated_at_ms = ?5
         WHERE command_id = ?1",
        params![
            command_id,
            execution_phase,
            next_version,
            serde_json::to_string(result)?,
            now_ms,
        ],
    )?;
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, ?2, ?3, ?4, 'none', 'control effect progress persisted', ?5, ?6)",
        params![
            command_id,
            next_version,
            phase,
            execution_phase,
            serde_json::to_string(result)?,
            now_ms,
        ],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        command_id,
        next_version,
        "command.progress",
        &json!({
            "command_id": command_id,
            "execution_mode": "control",
            "execution_phase": execution_phase,
            "terminal_status": "none",
            "projection_version": next_version,
            "result": result,
        }),
        now_ms,
    )?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn transition_business_command_for_task(
    root: &Path,
    task_id: &str,
    route_status: &str,
    result: Option<&Value>,
    error_code: Option<&str>,
    error_message: Option<&str>,
    reason: &str,
) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let tx = conn.transaction()?;
    let transitioned = transition_business_command_for_task_in_transaction(
        &tx,
        task_id,
        route_status,
        result,
        error_code,
        error_message,
        reason,
    )?;
    tx.commit()?;
    Ok(transitioned)
}

fn transition_business_command_for_task_in_transaction(
    tx: &Transaction<'_>,
    task_id: &str,
    route_status: &str,
    result: Option<&Value>,
    error_code: Option<&str>,
    error_message: Option<&str>,
    reason: &str,
) -> Result<bool> {
    let command_id = tx
        .query_row(
            "SELECT command_id FROM business_command_task_links WHERE task_id = ?1",
            params![task_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let Some(command_id) = command_id else {
        return Ok(false);
    };
    let (from_phase, prior_terminal, version) = tx.query_row(
        "SELECT execution_phase, terminal_status, projection_version
         FROM business_command_aggregates WHERE command_id = ?1",
        params![command_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, i64>(2)?,
            ))
        },
    )?;
    let normalized_route = canonical_queue_route_status(route_status)?;
    let persisted_route = if normalized_route == "running" {
        "leased"
    } else {
        normalized_route.as_str()
    };
    let (to_phase, terminal_status) = match normalized_route.as_str() {
        "handled" => ("terminal", "completed"),
        "failed" => ("terminal", "failed"),
        "cancelled" => ("terminal", "cancelled"),
        "leased" => ("leased", "none"),
        "running" => ("running", "none"),
        "blocked" => ("blocked", "none"),
        "pending"
            if matches!(
                from_phase.as_str(),
                "leased" | "running" | "awaiting_review" | "validating" | "retry_wait"
            ) =>
        {
            ("retry_wait", "none")
        }
        _ => ("queued", "none"),
    };
    if from_phase == "terminal" {
        anyhow::ensure!(
            terminal_status == prior_terminal || terminal_status == "none",
            "terminal command transition conflict for task `{task_id}`"
        );
        // lease-1 (F-002): a terminal command whose queue route is still
        // `leased`/`running` is an orphaned lease — the worker that owned it
        // disappeared and previously NO path cleared the row: this function
        // returned here without touching routing, so every lease sweep and
        // boot recovery reported the row released while it stayed `leased`
        // forever (acked_at NULL, zero workers, UI showing healthy progress).
        // Settle the route to match the durable terminal command so queue
        // state, command state, and UI projections agree. Idempotent: the
        // settled row is terminal and is no longer selected by lease sweeps,
        // and the already-terminal command is never re-queued or duplicated.
        let current_route = current_queue_route_status(tx, task_id)?;
        if matches!(current_route.as_str(), "leased" | "running") {
            let settled_route = match prior_terminal.as_str() {
                "completed" => "handled",
                "cancelled" => "cancelled",
                _ => "failed",
            };
            let settle_reason = format!(
                "orphaned queue lease settled to match terminal command ({prior_terminal}): {reason}"
            );
            set_routing_status(
                tx,
                task_id,
                settled_route,
                &now_iso_string(),
                "business-command-terminal-owner",
                &settle_reason,
                Some(settle_reason.as_str()),
            )?;
        }
        return Ok(true);
    }
    crate::business_os::command_lifecycle::validate_execution_phase_transition(
        &from_phase,
        to_phase,
    )?;
    if terminal_status == "completed" {
        let review_passed = tx.query_row(
            "SELECT EXISTS(
                SELECT 1 FROM business_command_results result_row
                JOIN business_command_aggregates aggregate_row
                  ON aggregate_row.command_id = result_row.command_id
                WHERE result_row.command_id = ?1
                  AND result_row.attempt = MAX(aggregate_row.attempt, 1)
                  AND result_row.review_status = 'passed'
                  AND result_row.validation_status = 'passed'
            )",
            params![command_id],
            |row| row.get::<_, bool>(0),
        )?;
        anyhow::ensure!(
            from_phase == "validating" && review_passed,
            "command completion requires persisted typed result plus passed review and validation"
        );
    }
    let now = now_iso_string();
    if persisted_route == "leased" {
        let owned_lease = tx
            .query_row(
                "SELECT lease_owner, leased_at, lease_expires_at
                 FROM communication_routing_state
                 WHERE message_key=?1 AND route_status='leased'",
                params![task_id],
                |row| {
                    Ok((
                        row.get::<_, Option<String>>(0)?,
                        row.get::<_, Option<String>>(1)?,
                        row.get::<_, Option<String>>(2)?,
                    ))
                },
            )
            .optional()?
            .is_some_and(|(owner, leased_at, expires_at)| {
                owner.is_some_and(|value| !value.trim().is_empty())
                    && leased_at.is_some_and(|value| !value.trim().is_empty())
                    && expires_at.is_some_and(|value| !value.trim().is_empty())
            });
        anyhow::ensure!(
            owned_lease,
            "business command `{command_id}` requires an owned, expiring queue lease before `{to_phase}`"
        );
    } else {
        set_routing_status(
            &tx,
            task_id,
            persisted_route,
            &now,
            "business-command-terminal-owner",
            reason,
            error_message.or_else(|| (normalized_route == "failed").then_some(reason)),
        )?;
    }
    let next_version = version.saturating_add(1);
    let now_ms = epoch_millis();
    let result_json = result.map(serde_json::to_string).transpose()?;
    tx.execute(
        "UPDATE business_command_aggregates
         SET execution_phase = ?2, terminal_status = ?3, projection_version = ?4,
             result_json = COALESCE(?5, result_json), error_code = ?6, error_message = ?7,
             retryable = CASE WHEN ?2 IN ('blocked', 'retry_wait') THEN 1 ELSE 0 END,
             attempt = attempt + CASE WHEN ?2 = 'running' AND execution_phase != 'running' THEN 1 ELSE 0 END,
             updated_at_ms = ?8
         WHERE command_id = ?1",
        params![
            command_id,
            to_phase,
            terminal_status,
            next_version,
            result_json,
            error_code,
            error_message,
            now_ms,
        ],
    )?;
    if to_phase == "terminal" {
        tx.execute(
            "UPDATE business_command_effects
             SET status = ?3, result_json = COALESCE(?4, result_json), error_message = ?5, updated_at_ms = ?6
             WHERE command_id = ?1 AND effect_key = ?2",
            params![
                command_id,
                format!("queue:{task_id}"),
                if terminal_status == "completed" { "completed" } else { "failed" },
                result_json,
                error_message,
                now_ms,
            ],
        )?;
    }
    let review_correlation = tx
        .query_row(
            "SELECT review_evidence_json FROM business_command_results
             WHERE command_id = ?1 ORDER BY attempt DESC LIMIT 1",
            params![command_id],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()?
        .flatten()
        .and_then(|raw| serde_json::from_str::<Value>(&raw).ok());
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        params![
            command_id,
            next_version,
            from_phase,
            to_phase,
            terminal_status,
            reason,
            serde_json::to_string(&json!({
                "task_id": task_id,
                "route_status": persisted_route,
                "result": result,
                "error_code": error_code,
                "error_message": error_message,
                "review_correlation": review_correlation,
            }))?,
            now_ms,
        ],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        &command_id,
        next_version,
        if to_phase == "terminal" {
            "command.terminal"
        } else {
            "command.progress"
        },
        &json!({
            "command_id": command_id,
            "execution_task_id": task_id,
            "execution_phase": to_phase,
            "terminal_status": terminal_status,
            "projection_version": next_version,
            "review_correlation": review_correlation,
        }),
        now_ms,
    )?;
    Ok(true)
}

pub(crate) fn persist_business_command_worker_result(
    root: &Path,
    task_id: &str,
    user_reply: &str,
) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let row = tx
        .query_row(
            "SELECT aggregate_row.command_id, aggregate_row.execution_phase,
                    aggregate_row.projection_version, MAX(aggregate_row.attempt, 1)
             FROM business_command_task_links link
             JOIN business_command_aggregates aggregate_row ON aggregate_row.command_id = link.command_id
             WHERE link.task_id = ?1",
            params![task_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, u32>(3)?,
                ))
            },
        )
        .optional()?;
    let Some((command_id, from_phase, version, attempt)) = row else {
        tx.commit()?;
        return Ok(false);
    };
    if from_phase == "terminal" {
        tx.commit()?;
        return Ok(true);
    }
    crate::business_os::command_lifecycle::validate_execution_phase_transition(
        &from_phase,
        "awaiting_review",
    )?;
    let existing = tx
        .query_row(
            "SELECT user_reply FROM business_command_results WHERE command_id = ?1 AND attempt = ?2",
            params![command_id, attempt],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    if let Some(existing) = existing {
        anyhow::ensure!(
            existing == user_reply,
            "worker result for command `{command_id}` attempt {attempt} is immutable"
        );
        tx.commit()?;
        return Ok(true);
    }
    let now_ms = epoch_millis();
    let result = json!({
        "command_id": command_id,
        "execution_task_id": task_id,
        "attempt": attempt,
        "status": "succeeded",
        "user_message": user_reply,
        // Compatibility alias for existing Business OS projections. The
        // canonical lifecycle-v2 field is `user_message`.
        "user_reply": user_reply,
        "structured_output": Value::Null,
        "artifacts": [],
        "writebacks": [],
        "verification_claims": [],
        "retry": Value::Null,
        "error": null,
    });
    tx.execute(
        "INSERT INTO business_command_results
            (command_id, attempt, status, user_reply, created_at_ms)
         VALUES (?1, ?2, 'succeeded', ?3, ?4)",
        params![command_id, attempt, user_reply, now_ms],
    )?;
    let next_version = version.saturating_add(1);
    tx.execute(
        "UPDATE business_command_aggregates
         SET execution_phase = 'awaiting_review', attempt = ?2, projection_version = ?3,
             result_json = ?4, updated_at_ms = ?5
         WHERE command_id = ?1",
        params![
            command_id,
            attempt,
            next_version,
            serde_json::to_string(&result)?,
            now_ms,
        ],
    )?;
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, ?2, ?3, 'awaiting_review', 'none', 'typed worker result persisted before review', ?4, ?5)",
        params![
            command_id,
            next_version,
            from_phase,
            serde_json::to_string(&json!({"task_id": task_id, "attempt": attempt}))?,
            now_ms,
        ],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        &command_id,
        next_version,
        "command.result_persisted",
        &json!({
            "command_id": command_id,
            "execution_task_id": task_id,
            "execution_phase": "awaiting_review",
            "projection_version": next_version,
        }),
        now_ms,
    )?;
    tx.commit()?;
    Ok(true)
}

pub(crate) fn record_business_command_review(
    root: &Path,
    task_id: &str,
    review_status: &str,
    validation_status: &str,
    evidence: &Value,
) -> Result<bool> {
    anyhow::ensure!(
        matches!(review_status, "passed" | "failed" | "held"),
        "invalid command review status"
    );
    anyhow::ensure!(
        matches!(validation_status, "passed" | "failed" | "pending"),
        "invalid command validation status"
    );
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let row = tx
        .query_row(
            "SELECT aggregate_row.command_id, aggregate_row.execution_phase,
                    aggregate_row.projection_version, MAX(aggregate_row.attempt, 1)
             FROM business_command_task_links link
             JOIN business_command_aggregates aggregate_row ON aggregate_row.command_id = link.command_id
             WHERE link.task_id = ?1",
            params![task_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, u32>(3)?,
                ))
            },
        )
        .optional()?;
    let Some((command_id, from_phase, version, attempt)) = row else {
        tx.commit()?;
        return Ok(false);
    };
    anyhow::ensure!(from_phase != "terminal", "cannot review a terminal command");
    let retryable_hold = review_status == "held"
        && evidence
            .get("retryable_hold")
            .and_then(Value::as_bool)
            .unwrap_or(false);
    let changed = tx.execute(
        "UPDATE business_command_results
         SET review_status = ?3, validation_status = ?4, review_evidence_json = ?5,
             reviewed_at_ms = ?6
         WHERE command_id = ?1 AND attempt = ?2",
        params![
            command_id,
            attempt,
            review_status,
            validation_status,
            serde_json::to_string(evidence)?,
            epoch_millis(),
        ],
    )?;
    anyhow::ensure!(
        changed == 1,
        "typed worker result is required before review"
    );
    let to_phase = if review_status == "passed" && validation_status == "passed" {
        "validating"
    } else if review_status == "failed" || validation_status == "failed" || retryable_hold {
        "retry_wait"
    } else {
        "blocked"
    };
    crate::business_os::command_lifecycle::validate_execution_phase_transition(
        &from_phase,
        to_phase,
    )?;
    let next_version = version.saturating_add(1);
    let now_ms = epoch_millis();
    tx.execute(
        "UPDATE business_command_aggregates
         SET execution_phase = ?2, projection_version = ?3,
             retryable = CASE WHEN ?2 IN ('retry_wait', 'blocked') THEN 1 ELSE 0 END,
             updated_at_ms = ?4
         WHERE command_id = ?1",
        params![command_id, to_phase, next_version, now_ms],
    )?;
    tx.execute(
        "INSERT INTO business_command_transitions
            (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
         VALUES (?1, ?2, ?3, ?4, 'none', 'completion review and validation recorded', ?5, ?6)",
        params![
            command_id,
            next_version,
            from_phase,
            to_phase,
            serde_json::to_string(evidence)?,
            now_ms,
        ],
    )?;
    insert_business_command_outbox_rows(
        &tx,
        &command_id,
        next_version,
        "command.reviewed",
        &json!({
            "command_id": command_id,
            "execution_task_id": task_id,
            "execution_phase": to_phase,
            "projection_version": next_version,
        }),
        now_ms,
    )?;
    tx.commit()?;
    Ok(true)
}

fn insert_business_command_outbox_rows(
    tx: &Transaction<'_>,
    command_id: &str,
    projection_version: i64,
    event_type: &str,
    payload: &Value,
    created_at_ms: i64,
) -> Result<()> {
    let payload_json = serde_json::to_string(payload)?;
    for destination in ["business-os", "rxdb"] {
        let event_id = format!("cmd-outbox:{command_id}:{projection_version}:{destination}");
        tx.execute(
            "INSERT OR IGNORE INTO business_command_outbox
                (event_id, command_id, projection_version, destination, event_type, payload_json,
                 status, attempts, next_attempt_at_ms, created_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'pending', 0, 0, ?7)",
            params![
                event_id,
                command_id,
                projection_version,
                destination,
                event_type,
                payload_json,
                created_at_ms,
            ],
        )?;
    }
    Ok(())
}

pub(crate) fn pending_business_command_outbox(
    root: &Path,
    limit: usize,
) -> Result<Vec<BusinessCommandOutboxEvent>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let now_ms = epoch_millis();
    let mut stmt = conn.prepare(
        "SELECT event_id, command_id, projection_version, destination, event_type, attempts
         FROM business_command_outbox
         WHERE status IN ('pending', 'failed') AND next_attempt_at_ms <= ?1
         ORDER BY created_at_ms ASC, event_id ASC
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![now_ms, limit.max(1) as i64], |row| {
        Ok(BusinessCommandOutboxEvent {
            event_id: row.get(0)?,
            command_id: row.get(1)?,
            projection_version: row.get(2)?,
            destination: row.get(3)?,
            event_type: row.get(4)?,
            attempts: row.get(5)?,
        })
    })?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(Into::into)
}

pub(crate) fn business_command_projection(root: &Path, command_id: &str) -> Result<Value> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let (
        module,
        command_type,
        record_id,
        payload_hash,
        execution_mode,
        execution_phase,
        terminal_status,
        attempt,
        projection_version,
        intent_json,
        result_json,
        error_code,
        error_message,
        retryable,
        created_at_ms,
        updated_at_ms,
    ) = conn.query_row(
        "SELECT module, command_type, record_id,
                payload_hash, execution_mode, execution_phase, terminal_status, attempt,
                projection_version, intent_json, result_json, error_code, error_message,
                retryable, created_at_ms, updated_at_ms
         FROM business_command_aggregates WHERE command_id = ?1",
        params![command_id],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
                row.get::<_, String>(4)?,
                row.get::<_, String>(5)?,
                row.get::<_, String>(6)?,
                row.get::<_, u32>(7)?,
                row.get::<_, i64>(8)?,
                row.get::<_, String>(9)?,
                row.get::<_, Option<String>>(10)?,
                row.get::<_, Option<String>>(11)?,
                row.get::<_, Option<String>>(12)?,
                row.get::<_, i64>(13)? != 0,
                row.get::<_, i64>(14)?,
                row.get::<_, i64>(15)?,
            ))
        },
    )?;
    let mut projection: Value = serde_json::from_str(&intent_json)?;
    anyhow::ensure!(
        projection.is_object(),
        "canonical command intent must be an object"
    );
    let task_id = conn
        .query_row(
            "SELECT task_id FROM business_command_task_links WHERE command_id = ?1",
            params![command_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .unwrap_or_default();
    let saga = conn
        .query_row(
            "SELECT saga_id, phase, current_step, total_steps, compensation_status
         FROM business_command_sagas WHERE command_id = ?1",
            params![command_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, i64>(2)?,
                    row.get::<_, i64>(3)?,
                    row.get::<_, String>(4)?,
                ))
            },
        )
        .optional()?;
    let status = if execution_phase == "terminal" {
        terminal_status.as_str()
    } else if execution_phase == "waiting_dependencies" {
        "waiting_dependencies"
    } else {
        "accepted"
    };
    let object = projection.as_object_mut().expect("object checked above");
    object.insert("id".to_string(), Value::String(command_id.to_string()));
    object.insert(
        "command_id".to_string(),
        Value::String(command_id.to_string()),
    );
    object.insert("module".to_string(), Value::String(module));
    object.insert("command_type".to_string(), Value::String(command_type));
    object.insert("record_id".to_string(), Value::String(record_id));
    object.insert("contract_version".to_string(), Value::from(2));
    object.insert("status".to_string(), Value::String(status.to_string()));
    object.insert(
        "replication_phase".to_string(),
        Value::String("native_observed".to_string()),
    );
    object.insert(
        "execution_mode".to_string(),
        Value::String(execution_mode.clone()),
    );
    object.insert(
        "execution_phase".to_string(),
        Value::String(execution_phase.clone()),
    );
    object.insert(
        "terminal_status".to_string(),
        Value::String(terminal_status.clone()),
    );
    object.insert(
        "execution_task_id".to_string(),
        Value::String(task_id.clone()),
    );
    object.insert("task_id".to_string(), Value::String(task_id));
    if let Some((saga_id, saga_phase, saga_step, saga_total_steps, compensation_status)) = saga {
        object.insert("saga_id".to_string(), Value::String(saga_id));
        object.insert("saga_phase".to_string(), Value::String(saga_phase));
        object.insert("saga_step".to_string(), Value::from(saga_step));
        object.insert(
            "saga_total_steps".to_string(),
            Value::from(saga_total_steps),
        );
        object.insert(
            "compensation_status".to_string(),
            Value::String(compensation_status),
        );
        if execution_phase != "terminal" {
            object.insert("pending_consistency".to_string(), Value::Bool(true));
        }
    }
    let (route_status, task_status) = if execution_phase == "terminal" {
        match terminal_status.as_str() {
            "completed" => ("handled", "completed"),
            "cancelled" => ("cancelled", "cancelled"),
            _ => ("failed", "failed"),
        }
    } else {
        match execution_phase.as_str() {
            "leased" | "running" | "awaiting_review" | "validating" => ("leased", "running"),
            "blocked" | "waiting_dependencies" => ("blocked", "blocked"),
            _ => ("pending", "queued"),
        }
    };
    object.insert(
        "route_status".to_string(),
        Value::String(route_status.to_string()),
    );
    object.insert(
        "task_status".to_string(),
        Value::String(task_status.to_string()),
    );
    object.insert("attempt".to_string(), Value::from(attempt));
    object.insert(
        "projection_version".to_string(),
        Value::from(projection_version),
    );
    object.insert("payload_hash".to_string(), Value::String(payload_hash));
    object.insert("retryable".to_string(), Value::Bool(retryable));
    object.insert("created_at_ms".to_string(), Value::from(created_at_ms));
    object.insert("updated_at_ms".to_string(), Value::from(updated_at_ms));
    if let Some(raw) = result_json {
        let result: Value = serde_json::from_str(&raw)?;
        if let Some(result_object) = result.as_object() {
            for field in ["outbound_text", "response", "answer"] {
                if let Some(value) = result_object.get(field) {
                    object.insert(field.to_string(), value.clone());
                }
            }
        }
        object.insert("result".to_string(), result);
    }
    if let Some(value) = error_code {
        object.insert("error_code".to_string(), Value::String(value));
    }
    if let Some(value) = error_message {
        object.insert("error_message".to_string(), Value::String(value));
    }
    Ok(projection)
}

pub(crate) fn inspect_business_command(root: &Path, command_id: &str) -> Result<Option<Value>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let exists = conn
        .query_row(
            "SELECT 1 FROM business_command_aggregates WHERE command_id = ?1",
            params![command_id],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    if !exists {
        return Ok(None);
    }
    let mut command = business_command_projection(root, command_id)?;
    redact_command_secrets(&mut command);
    let task_id = conn
        .query_row(
            "SELECT task_id FROM business_command_task_links WHERE command_id = ?1",
            params![command_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let mut transitions = Vec::new();
    let mut stmt = conn.prepare(
        "SELECT projection_version, from_phase, to_phase, terminal_status, reason,
                evidence_json, created_at_ms
         FROM business_command_transitions WHERE command_id = ?1
         ORDER BY projection_version ASC",
    )?;
    let rows = stmt.query_map(params![command_id], |row| {
        Ok(json!({
            "projection_version": row.get::<_, i64>(0)?,
            "from_phase": row.get::<_, String>(1)?,
            "to_phase": row.get::<_, String>(2)?,
            "terminal_status": row.get::<_, String>(3)?,
            "reason": row.get::<_, String>(4)?,
            "evidence": serde_json::from_str::<Value>(&row.get::<_, String>(5)?)
                .unwrap_or(Value::Null),
            "created_at_ms": row.get::<_, i64>(6)?,
        }))
    })?;
    for row in rows {
        transitions.push(row?);
    }
    let dependencies = command
        .pointer("/payload/dependencies")
        .cloned()
        .unwrap_or_else(|| json!([]));
    let attachments = command
        .pointer("/payload/attachments")
        .cloned()
        .unwrap_or_else(|| json!([]));
    Ok(Some(json!({
        "schema": "ctox.business_os.command_context.v1",
        "command": command,
        "execution_task_id": task_id,
        "dependencies": dependencies,
        "attachments": attachments,
        "transitions": transitions,
    })))
}

pub(crate) fn inspect_business_command_for_task(
    root: &Path,
    task_id: &str,
) -> Result<Option<Value>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let command_id = conn
        .query_row(
            "SELECT command_id FROM business_command_task_links WHERE task_id = ?1",
            params![task_id],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    command_id
        .as_deref()
        .map(|command_id| inspect_business_command(root, command_id))
        .transpose()
        .map(Option::flatten)
}

fn redact_command_secrets(value: &mut Value) {
    match value {
        Value::Object(object) => {
            for key in [
                "capability_token",
                "authorization",
                "access_token",
                "refresh_token",
                "secret",
            ] {
                if object.contains_key(key) {
                    object.insert(key.to_string(), Value::String("[REDACTED]".to_string()));
                }
            }
            for child in object.values_mut() {
                redact_command_secrets(child);
            }
        }
        Value::Array(items) => {
            for item in items {
                redact_command_secrets(item);
            }
        }
        _ => {}
    }
}

pub(crate) fn mark_business_command_outbox_delivered(root: &Path, event_id: &str) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    Ok(conn.execute(
        "UPDATE business_command_outbox
         SET status = 'delivered', delivered_at_ms = ?2, last_error = NULL
         WHERE event_id = ?1 AND status != 'delivered'",
        params![event_id, epoch_millis()],
    )? > 0)
}

pub(crate) fn mark_business_command_outbox_failed(
    root: &Path,
    event_id: &str,
    error: &str,
    max_attempts: u32,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let attempts = tx.query_row(
        "SELECT attempts + 1 FROM business_command_outbox WHERE event_id = ?1",
        params![event_id],
        |row| row.get::<_, u32>(0),
    )?;
    let dead_letter = attempts >= max_attempts.max(1);
    let backoff_ms = 250_i64.saturating_mul(1_i64 << attempts.min(8));
    tx.execute(
        "UPDATE business_command_outbox
         SET status = ?2, attempts = ?3, next_attempt_at_ms = ?4, last_error = ?5
         WHERE event_id = ?1",
        params![
            event_id,
            if dead_letter { "dead_letter" } else { "failed" },
            attempts,
            epoch_millis().saturating_add(backoff_ms),
            error,
        ],
    )?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn business_command_core_diagnostics(root: &Path) -> Result<Value> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let aggregate_count = conn.query_row(
        "SELECT COUNT(*) FROM business_command_aggregates",
        [],
        |row| row.get::<_, u64>(0),
    )?;
    let queue_commands_without_link = conn.query_row(
        "SELECT COUNT(*) FROM business_command_aggregates aggregate_row
         LEFT JOIN business_command_task_links link ON link.command_id = aggregate_row.command_id
         WHERE aggregate_row.execution_mode = 'queue'
           AND aggregate_row.execution_phase NOT IN ('waiting_dependencies', 'terminal')
           AND link.command_id IS NULL",
        [],
        |row| row.get::<_, u64>(0),
    )?;
    let links_without_task = conn.query_row(
        "SELECT COUNT(*) FROM business_command_task_links link
         LEFT JOIN communication_messages task ON task.message_key = link.task_id
         WHERE task.message_key IS NULL",
        [],
        |row| row.get::<_, u64>(0),
    )?;
    let (pending_outbox, dead_letter_outbox, oldest_pending_created_at_ms) = conn.query_row(
        "SELECT
            COALESCE(SUM(CASE WHEN status IN ('pending', 'failed') THEN 1 ELSE 0 END), 0),
            COALESCE(SUM(CASE WHEN status = 'dead_letter' THEN 1 ELSE 0 END), 0),
            MIN(CASE WHEN status IN ('pending', 'failed') THEN created_at_ms END)
         FROM business_command_outbox",
        [],
        |row| {
            Ok((
                row.get::<_, u64>(0)?,
                row.get::<_, u64>(1)?,
                row.get::<_, Option<i64>>(2)?,
            ))
        },
    )?;
    let uncertain_effects = conn.query_row(
        "SELECT COUNT(*) FROM business_command_effects WHERE status = 'uncertain'",
        [],
        |row| row.get::<_, u64>(0),
    )?;
    let (open_intake_failures, exhausted_intake_failures) = conn.query_row(
        "SELECT COUNT(*), COALESCE(SUM(CASE WHEN exhausted = 1 THEN 1 ELSE 0 END), 0)
         FROM business_command_intake_failures WHERE resolved_at_ms IS NULL",
        [],
        |row| Ok((row.get::<_, u64>(0)?, row.get::<_, u64>(1)?)),
    )?;
    let oldest_outbox_age_ms =
        oldest_pending_created_at_ms.map(|created| epoch_millis().saturating_sub(created).max(0));
    Ok(json!({
        "aggregate_count": aggregate_count,
        "queue_commands_without_link": queue_commands_without_link,
        "links_without_task": links_without_task,
        "orphan_link_count": queue_commands_without_link.saturating_add(links_without_task),
        "pending_outbox": pending_outbox,
        "dead_letter_outbox": dead_letter_outbox,
        "oldest_outbox_age_ms": oldest_outbox_age_ms,
        "uncertain_effects": uncertain_effects,
        "open_intake_failures": open_intake_failures,
        "exhausted_intake_failures": exhausted_intake_failures,
        "duplicate_effect_count": 0,
    }))
}

pub(crate) fn record_business_command_intake_failure(
    root: &Path,
    claim: BusinessCommandClaimRequest,
    error_message: &str,
    retry_budget: u32,
) -> Result<Value> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let tx = conn.transaction()?;
    let attempt = tx.query_row(
        "SELECT COALESCE(MAX(attempt), 0) + 1
         FROM business_command_intake_failures
         WHERE command_id = ?1 AND resolved_at_ms IS NULL",
        params![claim.command_id],
        |row| row.get::<_, u32>(0),
    )?;
    let exhausted = attempt >= retry_budget.max(1);
    let now_ms = epoch_millis();
    tx.execute(
        "INSERT INTO business_command_intake_failures
            (command_id, attempt, error_message, exhausted, observed_at_ms, resolved_at_ms)
         VALUES (?1, ?2, ?3, ?4, ?5, NULL)",
        params![
            claim.command_id,
            attempt,
            error_message,
            if exhausted { 1 } else { 0 },
            now_ms,
        ],
    )?;
    let canonical_exists = tx
        .query_row(
            "SELECT 1 FROM business_command_aggregates WHERE command_id = ?1",
            params![claim.command_id],
            |_| Ok(()),
        )
        .optional()?
        .is_some();
    let canonical_failure_created = exhausted && !canonical_exists;
    if canonical_failure_created {
        tx.execute(
            "INSERT INTO business_command_aggregates
                (command_id, idempotency_key, payload_hash, module, command_type, record_id,
                 execution_mode, execution_phase, terminal_status, attempt, projection_version,
                 intent_json, result_json, error_code, error_message, retryable,
                 created_at_ms, updated_at_ms)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, 'control', 'terminal', 'failed', ?7, 1,
                     ?8, ?9, 'native_unavailable', ?10, 0, ?11, ?12)",
            params![
                claim.command_id,
                claim.idempotency_key,
                claim.payload_hash,
                claim.module,
                claim.command_type,
                claim.record_id,
                attempt,
                serde_json::to_string(&claim.intent)?,
                serde_json::to_string(&json!({
                    "ok": false,
                    "error_code": "native_unavailable",
                    "error_message": error_message,
                }))?,
                error_message,
                claim.created_at_ms,
                now_ms,
            ],
        )?;
        tx.execute(
            "INSERT INTO business_command_transitions
                (command_id, projection_version, from_phase, to_phase, terminal_status, reason, evidence_json, created_at_ms)
             VALUES (?1, 1, 'native_observed', 'terminal', 'failed', 'native intake retry budget exhausted', ?2, ?3)",
            params![
                claim.command_id,
                serde_json::to_string(&json!({
                    "attempt": attempt,
                    "error_message": error_message,
                }))?,
                now_ms,
            ],
        )?;
        insert_business_command_outbox_rows(
            &tx,
            &claim.command_id,
            1,
            "command.intake_exhausted",
            &json!({
                "command_id": claim.command_id,
                "execution_phase": "terminal",
                "terminal_status": "failed",
                "projection_version": 1,
            }),
            now_ms,
        )?;
    }
    tx.commit()?;
    let failure_document = if canonical_failure_created {
        business_command_projection(root, &claim.command_id)?
    } else {
        claim.intent
    };
    Ok(json!({
        "command_id": claim.command_id,
        "attempt": attempt,
        "exhausted": exhausted,
        "canonical_exists": canonical_exists,
        "canonical_failure_created": canonical_failure_created,
        "failure_document": failure_document,
    }))
}

pub(crate) fn resolve_business_command_intake_failures(
    root: &Path,
    command_id: &str,
) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    Ok(conn.execute(
        "UPDATE business_command_intake_failures SET resolved_at_ms = ?2
         WHERE command_id = ?1 AND resolved_at_ms IS NULL",
        params![command_id, epoch_millis()],
    )?)
}

pub(crate) fn business_command_retention_maintenance(root: &Path, apply: bool) -> Result<Value> {
    const LARGE_RESULT_BYTES: usize = 64 * 1024;
    const DELIVERED_OUTBOX_RETENTION_MS: i64 = 30 * 24 * 60 * 60 * 1_000;
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let mut candidates = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT command_id, result_json FROM business_command_aggregates aggregate_row
             WHERE execution_phase = 'terminal'
               AND length(COALESCE(result_json, '')) > ?1
               AND NOT EXISTS (
                    SELECT 1 FROM business_command_outbox outbox
                    WHERE outbox.command_id = aggregate_row.command_id
                      AND outbox.status != 'delivered'
               )",
        )?;
        let rows = stmt.query_map(params![LARGE_RESULT_BYTES as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        for row in rows {
            candidates.push(row?);
        }
    }
    let artifact_root = root.join("runtime/business-command-artifacts");
    let mut externalized = 0_u64;
    if apply && !candidates.is_empty() {
        fs::create_dir_all(&artifact_root)?;
        let tx = conn.transaction()?;
        for (command_id, result_json) in &candidates {
            let digest = sha256_hex(result_json.as_bytes());
            let file_name = format!(
                "{}-{}.json",
                sanitize_path_component(command_id),
                &digest[..16]
            );
            let path = artifact_root.join(file_name);
            fs::write(&path, result_json)?;
            let reference = json!({
                "externalized": true,
                "artifact_ref": path.strip_prefix(root).unwrap_or(&path).display().to_string(),
                "sha256": digest,
                "size_bytes": result_json.len(),
            });
            tx.execute(
                "UPDATE business_command_aggregates SET result_json = ?2 WHERE command_id = ?1",
                params![command_id, serde_json::to_string(&reference)?],
            )?;
            externalized = externalized.saturating_add(1);
        }
        tx.commit()?;
    }
    let cutoff = epoch_millis().saturating_sub(DELIVERED_OUTBOX_RETENTION_MS);
    let delivered_outbox_candidates = conn.query_row(
        "SELECT COUNT(*) FROM business_command_outbox
         WHERE status = 'delivered' AND delivered_at_ms < ?1",
        params![cutoff],
        |row| row.get::<_, u64>(0),
    )?;
    let pruned_outbox = if apply {
        conn.execute(
            "DELETE FROM business_command_outbox
             WHERE status = 'delivered' AND delivered_at_ms < ?1",
            params![cutoff],
        )? as u64
    } else {
        0
    };
    Ok(json!({
        "apply": apply,
        "large_result_candidates": candidates.len(),
        "externalized_results": externalized,
        "delivered_outbox_candidates": delivered_outbox_candidates,
        "pruned_delivered_outbox": pruned_outbox,
        "aggregate_and_transition_evidence_deleted": 0,
        "policy": {
            "large_result_bytes": LARGE_RESULT_BYTES,
            "delivered_outbox_retention_ms": DELIVERED_OUTBOX_RETENTION_MS,
        },
    }))
}

pub(crate) fn reconcile_business_command_invariants(root: &Path, apply: bool) -> Result<Value> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let mut missing_links = Vec::new();
    let mut missing_tasks = Vec::new();
    let mut missing_outbox = Vec::new();
    let mut cancelled_queue_command_drift = Vec::new();
    {
        let mut stmt = conn.prepare(
            "SELECT aggregate_row.command_id, aggregate_row.execution_phase
             FROM business_command_aggregates aggregate_row
             LEFT JOIN business_command_task_links link ON link.command_id = aggregate_row.command_id
             WHERE aggregate_row.execution_mode = 'queue'
               AND aggregate_row.execution_phase NOT IN ('waiting_dependencies', 'terminal')
               AND link.command_id IS NULL",
        )?;
        for row in stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })? {
            let (command_id, phase) = row?;
            missing_links.push(json!({"command_id": command_id, "execution_phase": phase}));
        }
    }
    {
        let mut stmt = conn.prepare(
            "SELECT link.command_id, link.task_id FROM business_command_task_links link
             LEFT JOIN communication_messages task ON task.message_key = link.task_id
             WHERE task.message_key IS NULL",
        )?;
        for row in stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })? {
            let (command_id, task_id) = row?;
            missing_tasks.push(json!({"command_id": command_id, "execution_task_id": task_id}));
        }
    }
    {
        let mut stmt = conn.prepare(
            "SELECT aggregate_row.command_id, aggregate_row.projection_version
             FROM business_command_aggregates aggregate_row
             WHERE EXISTS (
                SELECT 1 FROM (SELECT 'business-os' AS destination UNION ALL SELECT 'rxdb') expected
                WHERE NOT EXISTS (
                    SELECT 1 FROM business_command_outbox outbox
                    WHERE outbox.command_id = aggregate_row.command_id
                      AND outbox.projection_version = aggregate_row.projection_version
                      AND outbox.destination = expected.destination
                )
             )",
        )?;
        for row in stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })? {
            let (command_id, version) = row?;
            missing_outbox.push((command_id, version));
        }
    }
    {
        let mut stmt = conn.prepare(
            "SELECT link.command_id, link.task_id, aggregate_row.execution_phase
             FROM business_command_task_links link
             JOIN business_command_aggregates aggregate_row
               ON aggregate_row.command_id = link.command_id
             JOIN communication_routing_state route
               ON route.message_key = link.task_id
             WHERE aggregate_row.execution_mode = 'queue'
               AND aggregate_row.execution_phase != 'terminal'
               AND route.route_status = 'cancelled'",
        )?;
        for row in stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })? {
            cancelled_queue_command_drift.push(row?);
        }
    }
    let mut repaired_cancelled_queue_commands = 0_u64;
    if apply && !cancelled_queue_command_drift.is_empty() {
        let tx = conn.transaction()?;
        for (_, task_id, _) in &cancelled_queue_command_drift {
            if transition_business_command_for_task_in_transaction(
                &tx,
                task_id,
                "cancelled",
                None,
                None,
                Some("queue task was already cancelled"),
                "reconciled cancelled queue task",
            )? {
                repaired_cancelled_queue_commands =
                    repaired_cancelled_queue_commands.saturating_add(1);
            }
        }
        tx.commit()?;
    }
    let mut repaired_outbox = 0_u64;
    if apply && !missing_outbox.is_empty() {
        let tx = conn.transaction()?;
        for (command_id, version) in &missing_outbox {
            insert_business_command_outbox_rows(
                &tx,
                command_id,
                *version,
                "command.reconciled",
                &json!({
                    "command_id": command_id,
                    "projection_version": version,
                    "reason": "missing current-version outbox repaired",
                }),
                epoch_millis(),
            )?;
            repaired_outbox = repaired_outbox.saturating_add(1);
        }
        tx.commit()?;
    }
    Ok(json!({
        "apply": apply,
        "missing_task_links": missing_links,
        "task_links_to_missing_tasks": missing_tasks,
        "missing_current_outbox": missing_outbox.iter().map(|(command_id, version)| json!({
            "command_id": command_id,
            "projection_version": version,
        })).collect::<Vec<_>>(),
        "cancelled_queue_command_drift": cancelled_queue_command_drift.iter().map(
            |(command_id, task_id, execution_phase)| json!({
                "command_id": command_id,
                "execution_task_id": task_id,
                "execution_phase": execution_phase,
                "queue_route_status": "cancelled",
            })
        ).collect::<Vec<_>>(),
        "repaired_outbox_commands": repaired_outbox,
        "repaired_cancelled_queue_commands": repaired_cancelled_queue_commands,
        "unsafe_repairs_applied": 0,
    }))
}

fn sanitize_path_component(value: &str) -> String {
    value
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || matches!(ch, '-' | '_') {
                ch
            } else {
                '_'
            }
        })
        .collect()
}

fn epoch_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or_default()
}

fn enforce_queue_task_spawn(
    conn: &Connection,
    metadata: &Value,
    parent_message_key: Option<&str>,
    thread_key: &str,
    message_key: &str,
    title: &str,
) -> Result<()> {
    let ticket_self_work_id = metadata_string_value(metadata, "ticket_self_work_id");
    let ticket_self_work_kind = metadata_string_value(metadata, "ticket_self_work_kind");
    let (parent_entity_type, parent_entity_id, spawn_kind, spawn_reason, budget_key, max_attempts) =
        if let Some(work_id) = ticket_self_work_id.clone() {
            (
                "WorkItem".to_string(),
                work_id.clone(),
                "self-work-queue-task".to_string(),
                "publish_self_work_for_execution".to_string(),
                format!("self-work-queue:{work_id}"),
                64,
            )
        } else if let Some(parent_message_key) = parent_message_key
            .map(str::trim)
            .filter(|value| !value.is_empty())
        {
            (
                "Message".to_string(),
                parent_message_key.to_string(),
                "queue-task".to_string(),
                "create_queue_task".to_string(),
                format!("queue-task:message:{parent_message_key}"),
                64,
            )
        } else if let Some(command_id) = metadata_string_value(metadata, "business_os_command_id") {
            (
                "ControlPlane".to_string(),
                format!("business-os-command:{command_id}"),
                "queue-task".to_string(),
                "create_business_os_command_queue_task".to_string(),
                format!("queue-task:business-os-command:{command_id}"),
                1,
            )
        } else {
            (
                "Thread".to_string(),
                thread_key.to_string(),
                "queue-task".to_string(),
                "create_queue_task".to_string(),
                format!("queue-task:thread:{thread_key}"),
                64,
            )
        };
    let mut edge_metadata = BTreeMap::new();
    edge_metadata.insert("thread_key".to_string(), thread_key.to_string());
    edge_metadata.insert("queue_title".to_string(), title.to_string());
    if let Some(skill) = metadata_string_value(metadata, "skill") {
        edge_metadata.insert("suggested_skill".to_string(), skill);
    }
    if let Some(workspace_root) = metadata_string_value(metadata, "workspace_root") {
        edge_metadata.insert("workspace_root".to_string(), workspace_root);
    }
    if let Some(run_class) = metadata_string_value(metadata, "core_run_class") {
        edge_metadata.insert("core_run_class".to_string(), run_class);
    }
    if let Some(kind) = ticket_self_work_kind {
        edge_metadata.insert("self_work_kind".to_string(), kind);
    }

    enforce_core_spawn_in_transaction(
        conn,
        &CoreSpawnRequest {
            parent_entity_type,
            parent_entity_id,
            child_entity_type: "QueueTask".to_string(),
            child_entity_id: message_key.to_string(),
            spawn_kind,
            spawn_reason,
            actor: "ctox-queue".to_string(),
            checkpoint_key: Some(message_key.to_string()),
            budget_key: Some(budget_key),
            max_attempts: Some(max_attempts),
            metadata: edge_metadata,
        },
    )?;
    Ok(())
}

fn metadata_string_value(metadata: &Value, key: &str) -> Option<String> {
    metadata
        .get(key)
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

fn merge_object_metadata(target: &mut Value, extra: Value) {
    let Some(target_map) = target.as_object_mut() else {
        return;
    };
    let Some(extra_map) = extra.as_object() else {
        return;
    };
    for (key, value) in extra_map {
        target_map.insert(key.clone(), value.clone());
    }
}

pub fn list_queue_tasks(
    root: &Path,
    statuses: &[String],
    limit: usize,
) -> Result<Vec<QueueTaskView>> {
    let db_path = resolve_db_path(root, None);
    let allowed = statuses
        .iter()
        .map(|status| status.trim().to_lowercase())
        .filter(|status| !status.is_empty())
        .collect::<Vec<_>>();
    if !statuses.is_empty() && allowed.is_empty() {
        return Ok(Vec::new());
    }
    let cache_key = queue_task_list_cache_key(&db_path, &allowed, limit);
    let stamp = queue_task_list_cache_stamp(&db_path);
    if let Some(tasks) = cached_queue_task_list(&cache_key, &stamp) {
        return Ok(tasks);
    }

    let conn = open_channel_db(&db_path)?;
    let tasks = if allowed.is_empty() {
        list_queue_tasks_from_conn(&conn, limit)?
    } else {
        list_queue_tasks_from_conn_with_statuses(&conn, &allowed, limit)?
    };
    drop(conn);
    let cache_key = queue_task_list_cache_key(&db_path, &allowed, limit);
    let stamp = queue_task_list_cache_stamp(&db_path);
    #[cfg(test)]
    record_queue_task_list_cache_miss_for_tests(&cache_key);
    store_queue_task_list_cache(cache_key, stamp, tasks.clone());
    Ok(tasks)
}

pub fn load_queue_task_for_business_os_command(
    root: &Path,
    command_id: &str,
) -> Result<Option<QueueTaskView>> {
    let command_id = command_id.trim();
    if command_id.is_empty() {
        return Ok(None);
    }
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    load_queue_task_for_business_os_command_from_conn(&conn, command_id)
}

pub fn count_queue_tasks(root: &Path, statuses: &[String]) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let allowed = statuses
        .iter()
        .map(|status| status.trim().to_lowercase())
        .filter(|status| !status.is_empty())
        .collect::<Vec<_>>();
    if !statuses.is_empty() && allowed.is_empty() {
        return Ok(0);
    }
    let cache_key = queue_task_count_cache_key(&db_path, &allowed);
    let stamp = queue_task_list_cache_stamp(&db_path);
    if let Some(count) = cached_queue_task_count(&cache_key, &stamp) {
        return Ok(count);
    }

    let conn = open_channel_db(&db_path)?;
    let count = if allowed.is_empty() {
        let count: i64 = conn.query_row(
            "SELECT COUNT(*)
             FROM communication_messages
             WHERE channel = ?1
               AND direction = 'inbound'",
            params![QUEUE_CHANNEL_NAME],
            |row| row.get(0),
        )?;
        count.max(0) as usize
    } else {
        count_queue_tasks_from_conn_with_statuses(&conn, &allowed)?
    };
    drop(conn);
    let cache_key = queue_task_count_cache_key(&db_path, &allowed);
    let stamp = queue_task_list_cache_stamp(&db_path);
    #[cfg(test)]
    record_queue_task_count_cache_miss_for_tests(&cache_key);
    store_queue_task_count_cache(cache_key, stamp, count);
    Ok(count)
}

pub(crate) fn pending_queue_task_count_uncached(root: &Path) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let Some(conn) = open_channel_db_read_only(&db_path)? else {
        return Ok(0);
    };
    if !channel_projection_tables_exist(
        &conn,
        &["communication_messages", "communication_routing_state"],
    )? {
        return Ok(0);
    }
    let count: i64 = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = ?1
          AND m.direction = 'inbound'
          AND lower(COALESCE(r.route_status, 'pending')) = 'pending'
          AND (
                json_extract(m.metadata_json, '$.not_before') IS NULL
             OR json_extract(m.metadata_json, '$.not_before') = ''
             OR json_extract(m.metadata_json, '$.not_before') <= strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
          )
        "#,
        params![QUEUE_CHANNEL_NAME],
        |row| row.get(0),
    )?;
    Ok(non_negative_i64_to_usize(count))
}

pub(crate) fn queue_task_deferred_until(root: &Path, message_key: &str) -> Result<Option<String>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    queue_task_deferred_until_from_conn(&conn, message_key)
}

fn queue_task_deferred_until_from_conn(
    conn: &Connection,
    message_key: &str,
) -> Result<Option<String>> {
    conn.query_row(
        r#"
        SELECT CASE
            WHEN COALESCE(json_extract(m.metadata_json, '$.not_before'), '') = ''
                THEN r.retry_not_before
            WHEN COALESCE(r.retry_not_before, '') = ''
                THEN json_extract(m.metadata_json, '$.not_before')
            WHEN datetime(r.retry_not_before) > datetime(json_extract(m.metadata_json, '$.not_before'))
                THEN r.retry_not_before
            ELSE json_extract(m.metadata_json, '$.not_before')
        END
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key=m.message_key
        WHERE m.message_key = ?1
          AND m.channel = ?2
          AND m.direction = 'inbound'
          AND (
                datetime(json_extract(m.metadata_json, '$.not_before')) > datetime('now')
             OR datetime(r.retry_not_before) > datetime('now')
          )
        LIMIT 1
        "#,
        params![message_key, QUEUE_CHANNEL_NAME],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map_err(anyhow::Error::from)
}

pub fn load_queue_task(root: &Path, message_key: &str) -> Result<Option<QueueTaskView>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    load_queue_task_from_conn(&conn, message_key)
}

pub fn load_queue_task_last_error(root: &Path, message_key: &str) -> Result<Option<String>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    conn.query_row(
        "SELECT NULLIF(TRIM(last_error), '')
         FROM communication_routing_state
         WHERE message_key = ?1
         LIMIT 1",
        params![message_key],
        |row| row.get::<_, Option<String>>(0),
    )
    .optional()
    .map(|value| value.flatten())
    .map_err(anyhow::Error::from)
}

pub fn update_queue_task(root: &Path, request: QueueTaskUpdateRequest) -> Result<QueueTaskView> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let current = load_queue_message_from_conn(&conn, &request.message_key)?
        .context("queue task not found")?;
    if request
        .route_status
        .as_deref()
        .is_some_and(|status| status.eq_ignore_ascii_case("pending"))
    {
        let command_state = conn
            .query_row(
                "SELECT aggregate.execution_phase, aggregate.terminal_status
                 FROM business_command_task_links link
                 JOIN business_command_aggregates aggregate
                   ON aggregate.command_id = link.command_id
                 WHERE link.task_id = ?1",
                params![request.message_key],
                |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
            )
            .optional()?;
        if let Some((phase, terminal_status)) = command_state {
            anyhow::ensure!(
                phase != "validating",
                "cannot release queue task `{}` while its Business OS command is validating; terminalization or an atomic completion hold must resolve it",
                request.message_key
            );
            anyhow::ensure!(
                phase != "terminal",
                "cannot release queue task `{}` for terminal Business OS command status `{terminal_status}`; submit a new command to retry",
                request.message_key
            );
        }
    }
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
    if let Some(workspace_root) = normalize_workspace_root(request.workspace_root.as_deref()) {
        metadata.insert("workspace_root".to_string(), Value::String(workspace_root));
    } else if request.clear_workspace_root {
        metadata.remove("workspace_root");
    } else if metadata
        .get("workspace_root")
        .and_then(Value::as_str)
        .and_then(|value| normalize_workspace_root(Some(value)))
        .is_none()
    {
        if let Some(workspace_root) = legacy_workspace_root_from_prompt(&prompt) {
            metadata.insert("workspace_root".to_string(), Value::String(workspace_root));
        }
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
    let releases_deferred_work = request
        .route_status
        .as_deref()
        .is_some_and(|status| status.eq_ignore_ascii_case("pending"));
    if releases_deferred_work {
        metadata.remove("not_before");
        metadata.remove("defer_reason");
    }
    let tx = conn.transaction()?;
    upsert_communication_message_tx(
        &tx,
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
        let status_note = request
            .status_note
            .as_deref()
            .map(str::trim)
            .filter(|value| !value.is_empty());
        let command_transitioned = matches!(
            route_status.trim().to_ascii_lowercase().as_str(),
            "pending" | "cancelled"
        ) && transition_business_command_for_task_in_transaction(
            &tx,
            &current.message_key,
            route_status,
            None,
            None,
            status_note,
            status_note.unwrap_or("queue task released for retry"),
        )?;
        if !command_transitioned {
            set_routing_status(
                &tx,
                &current.message_key,
                route_status,
                &now,
                "ctox-queue-update",
                "update_queue_task",
                status_note,
            )?;
        }
    }
    refresh_thread_tx(&tx, &thread_key)?;
    let updated = load_queue_task_from_conn(&tx, &current.message_key)?
        .context("failed to load updated queue task")?;
    tx.commit()?;
    Ok(updated)
}

pub fn set_queue_task_metadata_value(
    root: &Path,
    message_key: &str,
    key: &str,
    value: Value,
) -> Result<()> {
    let key = key.trim();
    anyhow::ensure!(!key.is_empty(), "queue metadata key must not be empty");
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let current =
        load_queue_message_from_conn(&conn, message_key)?.context("queue task not found")?;
    anyhow::ensure!(
        current.channel == QUEUE_CHANNEL_NAME && current.direction == "inbound",
        "message `{message_key}` is not a queue task"
    );
    let mut metadata = queue_metadata_object(&current.metadata);
    metadata.insert(key.to_string(), value);
    conn.execute(
        "UPDATE communication_messages
         SET metadata_json = ?2
         WHERE message_key = ?1
           AND channel = 'queue'
           AND direction = 'inbound'",
        params![message_key, serde_json::to_string(&metadata)?],
    )?;
    Ok(())
}

pub fn queue_task_metadata_value(
    root: &Path,
    message_key: &str,
    key: &str,
) -> Result<Option<Value>> {
    let key = key.trim();
    anyhow::ensure!(!key.is_empty(), "queue metadata key must not be empty");
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let Some(current) = load_queue_message_from_conn(&conn, message_key)? else {
        return Ok(None);
    };
    if current.channel != QUEUE_CHANNEL_NAME || current.direction != "inbound" {
        return Ok(None);
    }
    Ok(current.metadata.get(key).cloned())
}

pub fn lease_queue_task(
    root: &Path,
    message_key: &str,
    lease_owner: &str,
) -> Result<QueueTaskView> {
    let normalized_owner = lease_owner.trim();
    anyhow::ensure!(
        !normalized_owner.is_empty(),
        "lease owner must not be empty"
    );
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_queue_account(&mut conn)?;
    let current =
        load_queue_message_from_conn(&conn, message_key)?.context("queue task not found")?;
    // Thread refresh is ancillary routing maintenance. Run it before the lease
    // transaction so a transient write-lock error cannot be reported after the
    // lease has already committed and strand work between `pending` and worker
    // activation.
    refresh_thread(&mut conn, &current.thread_key)?;
    let now = now_iso_string();
    let lease_expires_at = (chrono::Utc::now() + chrono::Duration::minutes(15)).to_rfc3339();
    // Hold a write lock across the read-modify-write so a concurrent leaser on
    // a separate connection cannot overwrite our lease_owner (lost-update).
    let leased = {
        let tx =
            rusqlite::Transaction::new_unchecked(&conn, rusqlite::TransactionBehavior::Immediate)?;
        if let Some(not_before) = queue_task_deferred_until_from_conn(&tx, message_key)? {
            anyhow::bail!("queue task {message_key} is deferred until {not_before}");
        }
        let previous_route_status = current_queue_route_status(&tx, message_key)?;
        // Check-and-set: only lease a row that is free, ours already, or
        // pending. A losing racer flips 0 rows and must NOT load a task it
        // does not own.
        // Record the core-transition proof only after the CAS actually flips
        // the row, so a losing racer never writes a phantom proof.
        let updated = tx.execute(
            r#"INSERT INTO communication_routing_state (message_key, route_status, lease_owner, leased_at, first_pending_at, lease_expires_at, lease_worker_id, acked_at, last_error, updated_at)
               VALUES (?1, 'leased', ?2, ?3, ?3, ?4, NULL, NULL, NULL, ?3)
               ON CONFLICT(message_key) DO UPDATE SET route_status='leased', lease_owner=excluded.lease_owner, leased_at=excluded.leased_at, first_pending_at=COALESCE(communication_routing_state.first_pending_at, excluded.first_pending_at), lease_expires_at=excluded.lease_expires_at, lease_worker_id=NULL, retry_not_before=NULL, hold_reason=NULL, acked_at=NULL, updated_at=excluded.updated_at
               WHERE communication_routing_state.route_status = 'pending'"#,
            params![message_key, normalized_owner, now, lease_expires_at],
        )?;
        if updated != 0 {
            enforce_queue_route_status_transition(
                &tx,
                message_key,
                &previous_route_status,
                "leased",
                "ctox-queue-lease",
                "lease_queue_task",
            )?;
        } else {
            anyhow::bail!(
                "queue task {} lease lost: already leased by another owner",
                message_key
            );
        }
        let leased = load_queue_task_from_conn(&tx, message_key)?
            .context("failed to load leased queue task")?;
        tx.commit()?;
        leased
    };
    Ok(leased)
}

/// router-3: the age past which a still-leased, unacked queue task is "stuck".
/// Kept in lockstep with the process-mining `stuck_queue_items` diagnostic
/// (process_mining.rs) so the durable escalation and the advisory finding never
/// drift apart.
pub const STALE_QUEUE_LEASE_AGE_MINUTES: i64 = 15;

/// router-3: read-only — `message_key`s of queue-task leases held by `lease_owner`
/// that are older than [`STALE_QUEUE_LEASE_AGE_MINUTES`] and still unacked (the
/// exact condition process-mining flags as `stuck_queue_items`). This does NOT
/// release anything; it lets the reconciler escalate-as-evidence the in-active-key
/// case (a worker wedged mid-slice) that `release_stale_queue_task_leases`
/// intentionally protects from auto-release. The `leased_at < cutoff` comparison
/// mirrors the process-mining query (RFC3339-millis cutoff) so the two surfaces
/// agree on which leases are stuck.
pub fn list_stale_queue_task_leases(root: &Path, lease_owner: &str) -> Result<Vec<String>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let cutoff = (Utc::now() - Duration::minutes(STALE_QUEUE_LEASE_AGE_MINUTES))
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let mut statement = conn.prepare(
        r#"
        SELECT m.message_key
        FROM communication_messages m
        JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = 'queue'
          AND m.direction = 'inbound'
          AND r.route_status IN ('leased', 'running')
          AND r.lease_owner = ?1
          AND r.leased_at IS NOT NULL
          AND r.leased_at < ?2
          AND (r.acked_at IS NULL OR r.acked_at = '')
        ORDER BY r.leased_at ASC
        LIMIT 128
        "#,
    )?;
    let rows = statement.query_map(params![lease_owner, cutoff], |row| row.get::<_, String>(0))?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)
}

#[derive(Debug, Default)]
pub struct QueueLeaseSweepResult {
    pub released: Vec<String>,
    pub failures: Vec<String>,
}

pub fn release_stale_queue_task_leases(
    root: &Path,
    _lease_owner: &str,
    active_message_keys: &HashSet<String>,
) -> Result<QueueLeaseSweepResult> {
    #[derive(Debug)]
    struct Candidate {
        message_key: String,
        lease_owner: Option<String>,
        leased_at: Option<String>,
        lease_expires_at: Option<String>,
        lease_worker_id: Option<String>,
    }

    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    let mut statement = conn.prepare(
        r#"
        SELECT m.message_key, r.lease_owner, r.leased_at,
               r.lease_expires_at, r.lease_worker_id
        FROM communication_messages m
        JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.direction = 'inbound'
          AND r.route_status = 'leased'
          AND (
                r.lease_owner IS NULL
             OR trim(r.lease_owner) = ''
             OR r.leased_at IS NULL
             OR trim(r.leased_at) = ''
             OR r.lease_expires_at IS NULL
             OR trim(r.lease_expires_at) = ''
             OR datetime(r.lease_expires_at) <= datetime('now')
          )
        ORDER BY r.leased_at ASC, r.updated_at ASC
        LIMIT 128
        "#,
    )?;
    let rows = statement.query_map([], |row| {
        Ok(Candidate {
            message_key: row.get(0)?,
            lease_owner: row.get(1)?,
            leased_at: row.get(2)?,
            lease_expires_at: row.get(3)?,
            lease_worker_id: row.get(4)?,
        })
    })?;
    let candidates = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);

    let now = now_iso_string();
    let mut result = QueueLeaseSweepResult::default();
    for candidate in candidates {
        if active_message_keys.contains(&candidate.message_key) {
            continue;
        }
        // lease-2 (F-002): one bad candidate must not abort the whole sweep —
        // otherwise every orphaned lease queued behind a deterministically
        // failing row survives its worker forever. Each failing candidate is
        // retried on the next sweep pass; released rows stay released
        // (idempotent), so a re-run never duplicates the linked command.
        let outcome = (|| -> Result<bool> {
            let tx = conn.transaction()?;
            // Acquire SQLite's write lock while proving that every lease
            // identity field still matches the stale candidate. A renewed or
            // re-leased row changes at least one field and is left untouched.
            let claimed = tx.execute(
                r#"
                UPDATE communication_routing_state
                SET updated_at=updated_at
                WHERE message_key=?1
                  AND route_status='leased'
                  AND lease_owner IS ?2
                  AND leased_at IS ?3
                  AND lease_expires_at IS ?4
                  AND lease_worker_id IS ?5
                "#,
                params![
                    candidate.message_key,
                    candidate.lease_owner,
                    candidate.leased_at,
                    candidate.lease_expires_at,
                    candidate.lease_worker_id,
                ],
            )?;
            if claimed != 1 {
                tx.rollback()?;
                return Ok(false);
            }
            if transition_business_command_for_task_in_transaction(
                &tx,
                &candidate.message_key,
                "pending",
                None,
                None,
                Some("stale or ownerless queue lease recovered"),
                "stale or ownerless queue lease recovered",
            )? {
                tx.commit()?;
                return Ok(true);
            }
            enforce_queue_route_status_transition(
                &tx,
                &candidate.message_key,
                "leased",
                "pending",
                "ctox-communication-lease-repair",
                "release_stale_queue_task_leases",
            )?;
            let updated = tx.execute(
                r#"
            UPDATE communication_routing_state
            SET route_status='pending',
                lease_owner=NULL,
                leased_at=NULL,
                lease_expires_at=NULL,
                lease_worker_id=NULL,
                acked_at=NULL,
                last_error=NULL,
                updated_at=?2
            WHERE message_key = ?1
              AND route_status = 'leased'
              AND lease_owner IS ?3
              AND leased_at IS ?4
              AND lease_expires_at IS ?5
              AND lease_worker_id IS ?6
            "#,
                params![
                    candidate.message_key,
                    now,
                    candidate.lease_owner,
                    candidate.leased_at,
                    candidate.lease_expires_at,
                    candidate.lease_worker_id,
                ],
            )?;
            anyhow::ensure!(
                updated == 1,
                "stale lease identity changed before recovery write"
            );
            tx.commit()?;
            Ok(true)
        })();
        match outcome {
            Ok(true) => result.released.push(candidate.message_key),
            Ok(false) => {}
            Err(err) => result
                .failures
                .push(format!("{}: {err}", candidate.message_key)),
        }
    }
    Ok(result)
}

pub fn renew_message_leases(
    root: &Path,
    lease_owner: &str,
    message_keys: &[String],
) -> Result<usize> {
    if message_keys.is_empty() {
        return Ok(0);
    }
    let conn = open_channel_db(&resolve_db_path(root, None))?;
    let now = now_iso_string();
    let lease_expires_at = (chrono::Utc::now() + chrono::Duration::minutes(15)).to_rfc3339();
    let mut renewed = 0usize;
    for message_key in message_keys {
        renewed += conn.execute(
            r#"
            UPDATE communication_routing_state
            SET lease_expires_at=?3, updated_at=?4
            WHERE message_key=?1 AND route_status='leased' AND lease_owner=?2
            "#,
            params![message_key, lease_owner, lease_expires_at, now],
        )?;
    }
    Ok(renewed)
}

/// lease-3 (F-002): durable worker identity for queue-task leases. The worker
/// that actually starts a leased slice persists its instance-unique id on the
/// lease row so a recovery sweep (or operator) can tell a lease held by a
/// live worker apart from one left behind by a worker that disappeared. This
/// write is deliberately constrained to rows still leased by the expected
/// owner, so a stale worker can never stamp a lease that was already
/// reclaimed and re-leased by someone else.
pub fn record_queue_lease_worker(
    root: &Path,
    message_keys: &[String],
    lease_owner: &str,
    worker_id: &str,
) -> Result<usize> {
    let normalized_worker = worker_id.trim();
    if message_keys.is_empty() || normalized_worker.is_empty() {
        return Ok(0);
    }
    let conn = open_channel_db(&resolve_db_path(root, None))?;
    let now = now_iso_string();
    let mut updated = 0usize;
    for message_key in message_keys {
        updated += conn.execute(
            r#"
            UPDATE communication_routing_state
            SET lease_worker_id=?3, updated_at=?4
            WHERE message_key=?1 AND route_status='leased' AND lease_owner=?2
            "#,
            params![message_key, lease_owner, normalized_worker, now],
        )?;
    }
    Ok(updated)
}

/// lease-3 (F-002): read-only lease health probe for recovery sweeps and
/// status projections. A leased route is stalled when its lease is incomplete
/// (missing owner or timestamps) or its expiry has passed without a worker
/// heartbeat renewal — i.e. no live worker can still own it. Mirrors the
/// candidate condition in `release_stale_queue_task_leases`.
pub fn queue_task_lease_stalled(root: &Path, message_key: &str) -> Result<bool> {
    let conn = open_channel_db(&resolve_db_path(root, None))?;
    let stalled = conn.query_row(
        r#"
        SELECT EXISTS(
            SELECT 1 FROM communication_routing_state
            WHERE message_key = ?1
              AND route_status = 'leased'
              AND (
                    lease_owner IS NULL
                 OR trim(lease_owner) = ''
                 OR leased_at IS NULL
                 OR trim(leased_at) = ''
                 OR lease_expires_at IS NULL
                 OR trim(lease_expires_at) = ''
                 OR datetime(lease_expires_at) <= datetime('now')
              )
        )
        "#,
        params![message_key],
        |row| row.get::<_, bool>(0),
    )?;
    Ok(stalled)
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
    let mut edge_metadata = BTreeMap::new();
    edge_metadata.insert("thread_key".to_string(), thread_key.to_string());
    edge_metadata.insert("task_name".to_string(), task_name.to_string());
    edge_metadata.insert("scheduled_for".to_string(), scheduled_for.to_string());
    enforce_core_spawn(
        &conn,
        &CoreSpawnRequest {
            parent_entity_type: "ScheduleTask".to_string(),
            parent_entity_id: task_name.to_string(),
            child_entity_type: "Message".to_string(),
            child_entity_id: message_key.clone(),
            spawn_kind: "schedule-run-message".to_string(),
            spawn_reason: "emit_due_schedule".to_string(),
            actor: "ctox-schedule".to_string(),
            checkpoint_key: Some(run_id.to_string()),
            budget_key: Some(format!("schedule-run:{task_name}:{scheduled_for}")),
            max_attempts: Some(64),
            metadata: edge_metadata,
        },
    )?;
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
    let mut edge_metadata = BTreeMap::new();
    edge_metadata.insert("thread_key".to_string(), thread_key.to_string());
    edge_metadata.insert("goal_id".to_string(), goal_id.to_string());
    edge_metadata.insert("goal_title".to_string(), goal_title.to_string());
    edge_metadata.insert("step_title".to_string(), step_title.to_string());
    enforce_core_spawn(
        &conn,
        &CoreSpawnRequest {
            parent_entity_type: "PlanStep".to_string(),
            parent_entity_id: step_id.to_string(),
            child_entity_type: "Message".to_string(),
            child_entity_id: message_key.clone(),
            spawn_kind: "plan-step-message".to_string(),
            spawn_reason: "emit_plan_step".to_string(),
            actor: "ctox-plan".to_string(),
            checkpoint_key: Some(step_id.to_string()),
            budget_key: Some(format!("plan-step:{step_id}")),
            max_attempts: Some(8),
            metadata: edge_metadata,
        },
    )?;
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

/// Outcome of an IoT-event durable-work emission: which durable message key was
/// upserted (if any) and the spawn proof recorded in `ctox_core_spawn_edges`.
#[derive(Debug)]
pub struct IotEventEmitOutcome {
    /// The durable `communication_messages` key, present only when the spawn was
    /// accepted by the budget. Idempotent: the same `dedup_key` always maps to
    /// the same message key, so repeated matches collapse to ONE durable task.
    pub message_key: Option<String>,
    /// The recorded spawn-edge proof (accepted or budget-exhausted/rejected).
    pub spawn: CoreSpawnProof,
}

/// Emit ONE durable queue task for a matched IoT condition (§4A surface 3 /
/// §2A.15). This is the IoT analogue of `ingest_cron_message` /
/// `ingest_plan_message`: it is CTOX's mission brain doing the firing — there is
/// NO second automation engine in `iot::conditions`.
///
/// Boundedness comes from TWO complementary CTOX-native mechanisms (the plan's
/// "thin condition layer, not a second engine" decision):
///   * the durable `message_key` is derived from `dedup_key` and is the
///     `communication_messages` PRIMARY KEY, so re-firing the same condition
///     UPSERTs the same row — EXACTLY ONE durable queue task per dedup key
///     (§2A.15 re-trigger suppression, delegated to `queue.rs`/dedup), and
///   * every emission is a budget-bounded spawn edge under `budget_key`
///     (parent `IotAlarm` → child `QueueTask`, the registered
///     `iot-event-queue-task` contract family), so the *number of re-fires* is
///     provably bounded by the spawn budget recorded in `ctox_core_spawn_edges`
///     (§2A.20), not by a ported 100-trigger cap.
///
/// When the spawn budget is exhausted the durable message is NOT (re)written and
/// `message_key` is `None`; the caller treats that as suppressed re-firing.
#[allow(clippy::too_many_arguments)]
pub fn ingest_iot_event_message(
    root: &Path,
    alarm_id: &str,
    ruleset_id: &str,
    asset_id: &str,
    dedup_key: &str,
    budget_key: &str,
    max_attempts: i64,
    rule_name: &str,
    body: &str,
    skill: Option<&str>,
    observed_at: &str,
) -> Result<IotEventEmitOutcome> {
    let db_path = resolve_db_path(root, None);
    let mut conn = open_channel_db(&db_path)?;
    ensure_core_transition_guard_schema(&conn)?;
    ensure_account(
        &mut conn,
        "iot:system",
        "iot",
        "ctox IoT engine",
        "system",
        json!({"source": "iot"}),
    )?;

    // The durable task key IS the dedup key: the same matched condition for the
    // same asset/ruleset always resolves to the same `communication_messages`
    // PRIMARY KEY, so repeated matches UPSERT one row (one durable queue task).
    let message_key = format!("iot:system::{dedup_key}");
    let thread_key = format!("iot:{ruleset_id}:{asset_id}");
    let remote_id = format!("iot-{dedup_key}");

    // Budget-bounded spawn edge (parent IotAlarm → child QueueTask). The child
    // entity id is the per-fire alarm-scoped key so each accepted re-fire
    // consumes one unit of the finite budget recorded in ctox_core_spawn_edges.
    let child_id = format!("iot-qt:{alarm_id}");
    let mut edge_metadata = BTreeMap::new();
    edge_metadata.insert("iot_alarm_id".to_string(), alarm_id.to_string());
    edge_metadata.insert("iot_ruleset_id".to_string(), ruleset_id.to_string());
    edge_metadata.insert("iot_asset_id".to_string(), asset_id.to_string());
    edge_metadata.insert("dedup_key".to_string(), dedup_key.to_string());
    edge_metadata.insert("message_key".to_string(), message_key.clone());
    let spawn = evaluate_core_spawn(
        &conn,
        &CoreSpawnRequest {
            parent_entity_type: "IotAlarm".to_string(),
            parent_entity_id: alarm_id.to_string(),
            child_entity_type: "QueueTask".to_string(),
            child_entity_id: child_id,
            spawn_kind: "iot-event-queue-task".to_string(),
            spawn_reason: "iot_condition_match".to_string(),
            actor: "iot-conditions".to_string(),
            checkpoint_key: Some(alarm_id.to_string()),
            budget_key: Some(budget_key.to_string()),
            max_attempts: Some(max_attempts),
            metadata: edge_metadata,
        },
    )?;
    if !spawn.accepted {
        // Budget exhausted (or otherwise rejected): re-firing is bounded, so we
        // do NOT (re)write the durable task. One brain, finite budget.
        return Ok(IotEventEmitOutcome {
            message_key: None,
            spawn,
        });
    }

    let metadata = json!({
        "source": "ctox-iot",
        "iot_alarm_id": alarm_id,
        "iot_ruleset_id": ruleset_id,
        "iot_asset_id": asset_id,
        "dedup_key": dedup_key,
        "skill": skill,
    });
    upsert_communication_message(
        &mut conn,
        UpsertMessage {
            message_key: &message_key,
            channel: "iot",
            account_key: "iot:system",
            thread_key: &thread_key,
            remote_id: &remote_id,
            direction: "inbound",
            folder_hint: "iot",
            sender_display: "CTOX IoT engine",
            sender_address: "iot:system",
            recipient_addresses_json: "[]",
            cc_addresses_json: "[]",
            bcc_addresses_json: "[]",
            subject: rule_name,
            preview: &preview_text(body, rule_name),
            body_text: body,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "high",
            status: "received",
            seen: false,
            has_attachments: false,
            external_created_at: observed_at,
            observed_at,
            metadata_json: &serde_json::to_string(&metadata)?,
        },
    )?;
    refresh_thread(&mut conn, &thread_key)?;
    ensure_routing_rows_for_inbound(&conn)?;
    Ok(IotEventEmitOutcome {
        message_key: Some(message_key),
        spawn,
    })
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

#[derive(Debug)]
struct MessageAddressing {
    recipient_addresses: Vec<String>,
    cc_addresses: Vec<String>,
}

#[derive(Debug, Serialize)]
struct CommunicationStateCandidate {
    kind: String,
    message_key: String,
    channel: String,
    thread_key: String,
    created_at: String,
    summary: String,
}

#[derive(Debug, Serialize)]
struct CommunicationContextView {
    thread_key: String,
    latest_subject: Option<String>,
    latest_inbound: Option<CommunicationStateCandidate>,
    latest_outbound: Option<CommunicationStateCandidate>,
    thread_messages: Vec<ChannelMessageView>,
    related_messages: Vec<ChannelMessageView>,
    candidate_blockers: Vec<CommunicationStateCandidate>,
    candidate_promises: Vec<CommunicationStateCandidate>,
    open_owner_questions: Vec<CommunicationStateCandidate>,
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
    pub workspace_root: Option<String>,
    pub metadata: Value,
    pub preferred_reply_modality: Option<String>,
}

#[derive(Debug, Serialize)]
struct RoutingView {
    route_status: String,
    lease_owner: Option<String>,
    leased_at: Option<String>,
    acked_at: Option<String>,
    last_error: Option<String>,
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
    attachments: Vec<String>,
    sender_display: Option<String>,
    sender_address: Option<String>,
    send_voice: bool,
    reviewed_founder_send: bool,
}

#[derive(Debug, Clone, Default)]
pub(crate) struct FounderReplyAction {
    pub account_key: String,
    pub thread_key: String,
    pub subject: String,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub attachments: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct FounderOutboundAction {
    pub account_key: String,
    pub thread_key: String,
    pub subject: String,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub attachments: Vec<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub(crate) struct ExternalChatAction {
    pub channel: String,
    pub account_key: String,
    pub thread_key: String,
    pub subject: String,
    pub to: Vec<String>,
    pub cc: Vec<String>,
    pub attachments: Vec<String>,
}

fn sync_channel(root: &Path, db_path: &Path, channel: &str, args: &[String]) -> Result<Value> {
    let conn = open_channel_db(db_path)?;
    match communication_adapters::external_adapter_for_channel(channel) {
        Some(communication_adapters::ExternalCommunicationAdapter::Discord(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Email(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::GoogleChat(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Jami(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Matrix(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Mattermost(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Meeting(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Slack(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Teams(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Telegram(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Whatsapp(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        Some(communication_adapters::ExternalCommunicationAdapter::Zulip(adapter)) => {
            let adapter_json = adapter.sync_cli(
                root,
                &communication_adapters::AdapterSyncCommandRequest {
                    db_path,
                    passthrough_args: args,
                    skip_flags: &["--db", "--channel"],
                },
            )?;
            ensure_routing_rows_for_inbound(&conn)?;
            Ok(json!({
                "ok": true,
                "channel": adapter.channel_name(),
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        None => anyhow::bail!("unsupported channel sync target: {channel}"),
    }
}

fn send_message(root: &Path, db_path: &Path, request: ChannelSendRequest) -> Result<Value> {
    let mut conn = open_channel_db(db_path)?;
    let request = resolve_outbound_subject(&conn, request)?;
    enforce_external_chat_send_is_reviewed(&request)?;
    enforce_external_work_ack_has_pipeline_backing(&conn, &request)?;
    enforce_channel_attachment_support(&request)?;
    let reviewed_external_chat_approval =
        if request.reviewed_founder_send && is_reviewed_external_chat_channel(&request.channel) {
            let action = external_chat_action_from_send_request(&request);
            Some(require_any_unconsumed_external_chat_review(
                &conn,
                &action,
                &request.body,
            )?)
        } else {
            None
        };
    macro_rules! send_chat_adapter {
        ($factory:ident, $channel:literal) => {{
            let _core_send_proof = enforce_reviewed_communication_send_core_transition_if_approved(
                &conn,
                &request,
                reviewed_external_chat_approval.as_ref(),
            )?;
            let adapter = communication_adapters::$factory();
            let adapter_json = adapter.send_cli(
                root,
                &communication_adapters::ChatSendCommandRequest {
                    db_path,
                    account_key: &request.account_key,
                    thread_key: &request.thread_key,
                    to: &request.to,
                    cc: &request.cc,
                    sender_display: request.sender_display.as_deref(),
                    subject: &request.subject,
                    body: &request.body,
                    attachments: &request.attachments,
                },
            )?;
            if let Some((approval_key, _anchor_key)) = reviewed_external_chat_approval.as_ref() {
                mark_founder_reply_review_sent(&conn, approval_key, &adapter_json)?;
            }
            Ok(json!({
                "ok": true,
                "channel": $channel,
                "db_path": db_path,
                "status": adapter_json
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("sent"),
                "delivery_confirmed": adapter_json
                    .get("delivery")
                    .and_then(|value| value.get("confirmed"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "adapter_result": adapter_json,
            }))
        }};
    }
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
            let settings = runtime_settings_with_owner_profiles(
                root,
                communication_gateway::CommunicationAdapterKind::Email,
            );
            if request.reviewed_founder_send {
                let action = external_chat_action_from_send_request(&request);
                if let Ok((approval_key, _anchor_key)) =
                    require_any_unconsumed_external_chat_review(&conn, &action, &request.body)
                {
                    return send_reviewed_email_communication_request(
                        root,
                        &conn,
                        db_path,
                        &request,
                        &approval_key,
                    );
                }
                return send_reviewed_founder_outbound_request(root, &conn, db_path, &request);
            }
            validate_founder_outbound_email(&settings, &request)?;
            send_email_message(root, &conn, db_path, &request, None)
        }
        "discord" => send_chat_adapter!(discord, "discord"),
        "google_chat" => send_chat_adapter!(google_chat, "google_chat"),
        "jami" => {
            let _core_send_proof = enforce_reviewed_communication_send_core_transition_if_approved(
                &conn,
                &request,
                reviewed_external_chat_approval.as_ref(),
            )?;
            let adapter = communication_adapters::jami();
            let sender = request
                .sender_address
                .clone()
                .unwrap_or_else(|| jami_address_from_account_key(&request.account_key));
            let send_voice =
                request.send_voice || thread_prefers_voice_reply(&conn, &request.thread_key)?;
            let adapter_json = adapter.send_cli(
                root,
                &communication_adapters::JamiSendCommandRequest {
                    db_path,
                    account_id: &sender,
                    thread_key: &request.thread_key,
                    to: &request.to,
                    sender_display: request.sender_display.as_deref(),
                    subject: &request.subject,
                    body: &request.body,
                    send_voice,
                    attachments: &request.attachments,
                },
            )?;
            if let Some((approval_key, _anchor_key)) = reviewed_external_chat_approval.as_ref() {
                mark_founder_reply_review_sent(&conn, approval_key, &adapter_json)?;
            }
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
        "matrix" => send_chat_adapter!(matrix, "matrix"),
        "mattermost" => send_chat_adapter!(mattermost, "mattermost"),
        "teams" => {
            let _core_send_proof = enforce_reviewed_communication_send_core_transition_if_approved(
                &conn,
                &request,
                reviewed_external_chat_approval.as_ref(),
            )?;
            let adapter = communication_adapters::teams();
            let account_config = load_account_config(&conn, &request.account_key)?;
            let tenant_id = teams_tenant_from_account_config(account_config.as_ref());
            let adapter_json = adapter.send_cli(
                root,
                &communication_adapters::TeamsSendCommandRequest {
                    db_path,
                    tenant_id: tenant_id.as_deref().unwrap_or_default(),
                    thread_key: &request.thread_key,
                    to: &request.to,
                    sender_display: request.sender_display.as_deref(),
                    subject: &request.subject,
                    body: &request.body,
                    attachments: &request.attachments,
                },
            )?;
            if let Some((approval_key, _anchor_key)) = reviewed_external_chat_approval.as_ref() {
                mark_founder_reply_review_sent(&conn, approval_key, &adapter_json)?;
            }
            Ok(json!({
                "ok": true,
                "channel": "teams",
                "db_path": db_path,
                "status": adapter_json
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("sent"),
                "delivery_confirmed": adapter_json
                    .get("delivery")
                    .and_then(|value| value.get("confirmed"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "adapter_result": adapter_json,
            }))
        }
        "slack" => send_chat_adapter!(slack, "slack"),
        "telegram" => send_chat_adapter!(telegram, "telegram"),
        "whatsapp" => {
            let _core_send_proof = enforce_reviewed_communication_send_core_transition_if_approved(
                &conn,
                &request,
                reviewed_external_chat_approval.as_ref(),
            )?;
            let adapter = communication_adapters::whatsapp();
            let adapter_json = adapter.send_cli(
                root,
                &communication_adapters::WhatsappSendCommandRequest {
                    db_path,
                    account_key: &request.account_key,
                    thread_key: &request.thread_key,
                    to: &request.to,
                    sender_display: request.sender_display.as_deref(),
                    body: &request.body,
                    attachments: &request.attachments,
                },
            )?;
            if let Some((approval_key, _anchor_key)) = reviewed_external_chat_approval.as_ref() {
                mark_founder_reply_review_sent(&conn, approval_key, &adapter_json)?;
            }
            Ok(json!({
                "ok": true,
                "channel": "whatsapp",
                "db_path": db_path,
                "status": adapter_json
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("sent"),
                "delivery_confirmed": adapter_json
                    .get("delivery")
                    .and_then(|value| value.get("confirmed"))
                    .and_then(Value::as_bool)
                    .unwrap_or(false),
                "adapter_result": adapter_json,
            }))
        }
        "zulip" => send_chat_adapter!(zulip, "zulip"),
        "meeting" => {
            let _core_send_proof = enforce_reviewed_communication_send_core_transition_if_approved(
                &conn,
                &request,
                reviewed_external_chat_approval.as_ref(),
            )?;
            let adapter = communication_adapters::meeting();
            let session_id = &request.thread_key;
            let adapter_json = adapter.send_cli(
                root,
                &communication_adapters::MeetingSendCommandRequest {
                    db_path,
                    session_id,
                    body: &request.body,
                },
            )?;
            if let Some((approval_key, _anchor_key)) = reviewed_external_chat_approval.as_ref() {
                mark_founder_reply_review_sent(&conn, approval_key, &adapter_json)?;
            }
            Ok(json!({
                "ok": true,
                "channel": "meeting",
                "db_path": db_path,
                "status": adapter_json
                    .get("status")
                    .and_then(Value::as_str)
                    .unwrap_or("sent"),
                "adapter_result": adapter_json,
            }))
        }
        other => anyhow::bail!("unsupported channel send target: {other}"),
    }
}

fn enforce_channel_attachment_support(request: &ChannelSendRequest) -> Result<()> {
    if request.attachments.is_empty() {
        return Ok(());
    }
    if chat_channel_is_text_only_v1(&request.channel) {
        anyhow::bail!(
            "{} attachments are not supported by the native chat adapter v1. Send a text-only message or upload/share the file through a supported channel after attachment handling has a provider-specific security review.",
            request.channel
        );
    }
    Ok(())
}

fn chat_channel_is_text_only_v1(channel: &str) -> bool {
    matches!(
        channel,
        "slack" | "discord" | "telegram" | "matrix" | "mattermost" | "zulip" | "google_chat"
    )
}

fn is_review_required_outbound_channel(channel: &str) -> bool {
    matches!(
        channel,
        "email"
            | "teams"
            | "jami"
            | "whatsapp"
            | "meeting"
            | "slack"
            | "discord"
            | "telegram"
            | "matrix"
            | "mattermost"
            | "zulip"
            | "google_chat"
    )
}

fn enforce_external_chat_send_is_reviewed(request: &ChannelSendRequest) -> Result<()> {
    if is_review_required_outbound_channel(&request.channel) && !request.reviewed_founder_send {
        anyhow::bail!(
            "outbound {} communication must pass communication review before sending. Draft the response for completion review first; after approval the Harness sends the exact approved body through the reviewed send path.",
            request.channel
        );
    }
    Ok(())
}

fn enforce_external_work_ack_has_pipeline_backing(
    conn: &Connection,
    request: &ChannelSendRequest,
) -> Result<()> {
    if !matches!(
        request.channel.as_str(),
        "teams"
            | "jami"
            | "whatsapp"
            | "meeting"
            | "slack"
            | "discord"
            | "telegram"
            | "matrix"
            | "mattermost"
            | "zulip"
            | "google_chat"
    ) {
        return Ok(());
    }
    if !body_promises_follow_up_work(&request.body) {
        return Ok(());
    }
    if thread_has_open_work_backing(conn, &request.thread_key)? {
        if request.reviewed_founder_send {
            return Ok(());
        }
        anyhow::bail!(
            "outbound {} acknowledgement promises follow-up work and has pipeline backing, but it has not passed communication review. Draft the quick response for review first; after approval the Harness sends the exact approved body.",
            request.channel
        );
    }
    anyhow::bail!(
        "outbound {} acknowledgement promises follow-up work but no durable queue item, plan, or internal work item exists for thread `{}`. Create the pipeline item first, then send the acknowledgement.",
        request.channel,
        request.thread_key
    )
}

fn body_promises_follow_up_work(body: &str) -> bool {
    let normalized = format!(
        "{} {}",
        body.to_lowercase(),
        normalize_deliverable_text(body)
    );
    text_mentions_any(
        &normalized,
        &[
            "ich scrolle",
            "ich uebertrage",
            "ich übertrage",
            "ich erstelle",
            "ich bearbeite",
            "ich kuemmere",
            "ich kümmere",
            "ich pruefe",
            "ich prüfe",
            "ich recherchiere",
            "ich lese",
            "ich extrahiere",
            "ich sende",
            "ich melde",
            "ich mache",
            "ich werde",
            "werde ich",
            "i will",
            "i ll",
            "i am going to",
            "i will check",
            "i will create",
            "i will send",
            "working on it",
        ],
    )
}

fn thread_has_open_work_backing(conn: &Connection, thread_key: &str) -> Result<bool> {
    if open_queue_backing_exists(conn, thread_key)? {
        return Ok(true);
    }
    if table_exists(conn, "planned_goals")? && open_plan_backing_exists(conn, thread_key)? {
        return Ok(true);
    }
    if table_exists(conn, "ticket_self_work_items")?
        && open_self_work_backing_exists(conn, thread_key)?
    {
        return Ok(true);
    }
    Ok(false)
}

fn open_queue_backing_exists(conn: &Connection, thread_key: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = 'queue'
          AND m.direction = 'inbound'
          AND m.thread_key = ?1
          AND COALESCE(r.route_status, 'pending') NOT IN ('handled', 'cancelled', 'failed', 'superseded')
        "#,
        params![thread_key],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn open_plan_backing_exists(conn: &Connection, thread_key: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM planned_goals
        WHERE thread_key = ?1
          AND status NOT IN ('completed', 'closed', 'cancelled', 'failed', 'superseded')
        "#,
        params![thread_key],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn open_self_work_backing_exists(conn: &Connection, thread_key: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM ticket_self_work_items
        WHERE state NOT IN ('closed', 'cancelled', 'failed', 'superseded', 'blocked')
          AND (
            json_extract(metadata_json, '$.thread_key') = ?1
            OR json_extract(metadata_json, '$.parent_thread_key') = ?1
            OR body_text LIKE '%' || ?1 || '%'
          )
        "#,
        params![thread_key],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn table_exists(conn: &Connection, table_name: &str) -> Result<bool> {
    conn.query_row(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1 LIMIT 1",
        params![table_name],
        |_| Ok(true),
    )
    .optional()
    .map(|value| value.unwrap_or(false))
    .map_err(anyhow::Error::from)
}

fn load_message_from_conn(
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
            r.last_error,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.message_key = ?1
        LIMIT 1
        "#,
        params![message_key],
        map_channel_message_row,
    )
    .optional()
    .map_err(anyhow::Error::from)
}

fn load_message_addressing_from_conn(
    conn: &Connection,
    message_key: &str,
) -> Result<Option<MessageAddressing>> {
    conn.query_row(
        r#"
        SELECT recipient_addresses_json, cc_addresses_json
        FROM communication_messages
        WHERE message_key = ?1
        LIMIT 1
        "#,
        params![message_key],
        |row| {
            Ok(MessageAddressing {
                recipient_addresses: parse_string_json_array(&row.get::<_, String>(0)?),
                cc_addresses: parse_string_json_array(&row.get::<_, String>(1)?),
            })
        },
    )
    .optional()
    .map_err(anyhow::Error::from)
}

fn normalize_email_list(values: impl IntoIterator<Item = String>) -> Vec<String> {
    let mut seen = BTreeSet::new();
    let mut ordered = Vec::new();
    for value in values {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            continue;
        }
        let normalized = normalize_email_address(trimmed);
        if normalized.is_empty() || !seen.insert(normalized.clone()) {
            continue;
        }
        ordered.push(trimmed.to_string());
    }
    ordered
}

fn normalize_deliverable_text(value: &str) -> String {
    value
        .to_ascii_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch.is_ascii_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
}

fn text_mentions_any(haystack: &str, needles: &[&str]) -> bool {
    needles.iter().any(|needle| haystack.contains(needle))
}

fn detect_required_founder_deliverables(subject: &str, body: &str) -> Vec<String> {
    let normalized = format!(
        "{} {}",
        normalize_deliverable_text(subject),
        normalize_deliverable_text(body)
    );
    let mut required = Vec::new();
    if text_mentions_any(&normalized, &["qr code", "qrcode", "jami qr", "qr zugang"]) {
        required.push("qr_code".to_string());
    }
    if text_mentions_any(
        &normalized,
        &[
            "5 mockups",
            "fuenf mockups",
            "fuenf verschiedenen design vorlagen",
            "5 verschiedenen design vorlagen",
            "mockups",
            "entwuerfe",
            "entwurfe",
            "standalone html mockup",
        ],
    ) {
        required.push("mockup_links_or_files".to_string());
    }
    if text_mentions_any(
        &normalized,
        &[
            "link set",
            "linkset",
            "links schicken",
            "schick links",
            "verlinkten zwischenstand",
            "oeffentlichen links",
            "offentlichen links",
        ],
    ) {
        required.push("link_set".to_string());
    }
    normalize_email_list(required)
}

fn attachments_satisfy_deliverable(attachments: &[String], deliverable: &str) -> bool {
    let lowered = attachments
        .iter()
        .map(|value| value.to_ascii_lowercase())
        .collect::<Vec<_>>();
    match deliverable {
        "qr_code" => lowered.iter().any(|value| {
            (value.contains("jami") || value.contains("qr")) && value.ends_with(".pdf")
        }),
        "mockup_links_or_files" => lowered.iter().any(|value| {
            value.ends_with(".html") || value.ends_with(".pdf") || value.ends_with(".png")
        }),
        "link_set" => false,
        _ => false,
    }
}

fn founder_reply_satisfies_deliverable(
    body: &str,
    attachments: &[String],
    deliverable: &str,
) -> bool {
    if attachments_satisfy_deliverable(attachments, deliverable) {
        return true;
    }
    let normalized = normalize_deliverable_text(body);
    match deliverable {
        "qr_code" => text_mentions_any(&normalized, &["qr code", "qrcode", "jami qr", "qr zugang"]),
        "mockup_links_or_files" => text_mentions_any(
            &normalized,
            &[
                "mockup",
                "entwurf",
                "design vorlage",
                "html",
                "http",
                "https",
                "link",
            ],
        ),
        "link_set" => text_mentions_any(&normalized, &["http", "https", "link", "links"]),
        _ => true,
    }
}

fn prepare_founder_reply_attachments(
    root: &Path,
    subject: &str,
    body: &str,
) -> Result<Vec<String>> {
    let required = detect_required_founder_deliverables(subject, body);
    let mut attachments = Vec::new();
    if required.iter().any(|value| value == "qr_code")
        && normalize_deliverable_text(&format!("{subject} {body}")).contains("jami")
    {
        attachments.push(generate_jami_setup_pdf_artifact(root)?);
    }
    Ok(attachments)
}

fn generate_jami_setup_pdf_artifact(root: &Path) -> Result<String> {
    let settings = communication_gateway::runtime_settings_from_root(
        root,
        communication_gateway::CommunicationAdapterKind::Jami,
    );
    let account_id = settings
        .get("CTO_JAMI_ACCOUNT_ID")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .context("missing CTO_JAMI_ACCOUNT_ID for Jami QR artifact generation")?;
    let profile_name = settings
        .get("CTO_JAMI_PROFILE_NAME")
        .map(|value| value.trim())
        .filter(|value| !value.is_empty())
        .unwrap_or("CTO1");
    let share_uri = format!("jami:{account_id}");
    let artifact_dir = root.join("runtime/communication/artifacts/jami");
    fs::create_dir_all(&artifact_dir).with_context(|| {
        format!(
            "failed to create Jami artifact dir {}",
            artifact_dir.display()
        )
    })?;
    let file_name = format!(
        "ctox-jami-setup-{}.pdf",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let path = artifact_dir.join(file_name);
    let bytes = build_simple_jami_setup_pdf(profile_name, &share_uri)?;
    fs::write(&path, bytes)
        .with_context(|| format!("failed to write Jami setup PDF {}", path.display()))?;
    Ok(path.display().to_string())
}

fn build_simple_jami_setup_pdf(profile_name: &str, share_uri: &str) -> Result<Vec<u8>> {
    let qr = QrCode::new(share_uri.as_bytes()).context("failed to build Jami QR code")?;
    let width = qr.width();
    let colors = qr.to_colors();
    let mut content = String::new();
    content.push_str("BT /F1 20 Tf 72 760 Td ");
    content.push_str(&pdf_text(profile_name));
    content.push_str(" Tj ET\n");
    content.push_str("BT /F1 12 Tf 72 738 Td ");
    content.push_str(&pdf_text("Scan this QR code in Jami or use the URI below."));
    content.push_str(" Tj ET\n");
    content.push_str("BT /F1 11 Tf 72 718 Td ");
    content.push_str(&pdf_text(share_uri));
    content.push_str(" Tj ET\n");
    content.push_str("0 0 0 rg\n");
    let module = 5.0f32;
    let origin_x = 72.0f32;
    let origin_y = 420.0f32;
    for y in 0..width {
        for x in 0..width {
            let idx = y * width + x;
            if matches!(colors.get(idx), Some(QrColor::Dark)) {
                let px = origin_x + (x as f32 * module);
                let py = origin_y + ((width - 1 - y) as f32 * module);
                content.push_str(&format!("{px:.2} {py:.2} {module:.2} {module:.2} re f\n"));
            }
        }
    }
    content.push_str("BT /F1 10 Tf 72 396 Td ");
    content.push_str(&pdf_text("Account name:"));
    content.push_str(" Tj ET\n");
    content.push_str("BT /F1 10 Tf 140 396 Td ");
    content.push_str(&pdf_text(profile_name));
    content.push_str(" Tj ET\n");
    content.push_str("BT /F1 10 Tf 72 380 Td ");
    content.push_str(&pdf_text("Fallback URI:"));
    content.push_str(" Tj ET\n");
    content.push_str("BT /F1 10 Tf 140 380 Td ");
    content.push_str(&pdf_text(share_uri));
    content.push_str(" Tj ET\n");

    let mut objects = Vec::new();
    objects.push("1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n".to_string());
    objects.push("2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n".to_string());
    objects.push("3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n".to_string());
    objects.push(
        "4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n".to_string(),
    );
    objects.push(format!(
        "5 0 obj << /Length {} >> stream\n{}endstream\nendobj\n",
        content.as_bytes().len(),
        content
    ));

    let mut pdf = b"%PDF-1.4\n".to_vec();
    let mut offsets = vec![0usize];
    for object in &objects {
        offsets.push(pdf.len());
        pdf.extend_from_slice(object.as_bytes());
    }
    let xref_start = pdf.len();
    pdf.extend_from_slice(format!("xref\n0 {}\n", offsets.len()).as_bytes());
    pdf.extend_from_slice(b"0000000000 65535 f \n");
    for offset in offsets.iter().skip(1) {
        pdf.extend_from_slice(format!("{offset:010} 00000 n \n").as_bytes());
    }
    pdf.extend_from_slice(
        format!(
            "trailer << /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n",
            offsets.len(),
            xref_start
        )
        .as_bytes(),
    );
    Ok(pdf)
}

fn pdf_text(value: &str) -> String {
    let escaped = value
        .replace('\\', "\\\\")
        .replace('(', "\\(")
        .replace(')', "\\)");
    format!("({escaped})")
}

fn derives_targets_from_forward(subject: &str, body: &str) -> bool {
    let lowered_subject = subject.to_ascii_lowercase();
    if lowered_subject.starts_with("fwd:") || lowered_subject.starts_with("fw:") {
        return true;
    }
    let lowered_body = body.to_ascii_lowercase();
    lowered_body.contains("weitergeleiteten nachricht")
        || lowered_body.contains("begin forwarded message")
        || lowered_body.contains("forwarded message")
}

fn derive_founder_reply_recipients(
    inbound: &ChannelMessageView,
    addressing: &MessageAddressing,
) -> (Vec<String>, Vec<String>) {
    let account_email =
        normalize_email_address(&email_address_from_account_key(&inbound.account_key));
    let sender_email = normalize_email_address(&inbound.sender_address);

    let filter_external = |values: &[String]| {
        values
            .iter()
            .filter(|value| {
                let normalized = normalize_email_address(value);
                !normalized.is_empty() && normalized != account_email && normalized != sender_email
            })
            .cloned()
            .collect::<Vec<_>>()
    };

    let external_to = normalize_email_list(filter_external(&addressing.recipient_addresses));
    let external_cc = normalize_email_list(filter_external(&addressing.cc_addresses));

    if derives_targets_from_forward(&inbound.subject, &inbound.body_text) && !external_to.is_empty()
    {
        let mut cc = vec![inbound.sender_address.clone()];
        cc.extend(external_cc);
        return (external_to, normalize_email_list(cc));
    }

    let mut cc = external_to;
    cc.extend(external_cc);
    (
        vec![inbound.sender_address.clone()],
        normalize_email_list(cc),
    )
}

fn protected_recipient_policies(
    settings: &BTreeMap<String, String>,
    request: &ChannelSendRequest,
) -> Vec<EmailSenderPolicy> {
    request
        .to
        .iter()
        .chain(request.cc.iter())
        .map(|email| classify_email_sender(settings, email))
        .filter(|policy| matches!(policy.role.as_str(), "owner" | "founder" | "admin"))
        .collect::<Vec<_>>()
}

fn ensure_founder_outbound_body_clean(request: &ChannelSendRequest) -> Result<()> {
    ensure_founder_outbound_body_text_clean(&request.body)
}

pub(crate) fn ensure_founder_outbound_body_text_clean(body: &str) -> Result<()> {
    let lowered = body.to_ascii_lowercase();
    let first_lines = body
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(5)
        .collect::<Vec<_>>();
    let header_preamble_hits = first_lines
        .iter()
        .filter(|line| {
            let lowered = line.to_ascii_lowercase();
            lowered.starts_with("an:")
                || lowered.starts_with("to:")
                || lowered.starts_with("cc:")
                || lowered.starts_with("bcc:")
                || lowered.starts_with("betreff:")
                || lowered.starts_with("subject:")
        })
        .copied()
        .collect::<Vec<_>>();
    if !header_preamble_hits.is_empty() {
        anyhow::bail!(
            "founder/owner outbound email failed communication review because addressing or subject headers were placed in the message body: {}",
            header_preamble_hits.join(", ")
        );
    }
    let forbidden_markers = [
        "/home/",
        "queue:",
        "runtime/ctox.sqlite3",
        "strategic direction setup",
        "review rework",
        "review-rework",
        "self-work",
        "thread_key",
        "message_key",
        "conversation_id",
        "lease_owner",
        "route_status",
        "routing-state",
        "review-approval",
        "review approval",
        "send-proof",
        "send proof",
        "outbound-message-row",
        "outbound message row",
        "review/send proof",
        "inbound `email:",
        "steht jetzt auf `handled`",
        "status `handled`",
        "sqlite",
        "host-pfad",
        "host-pfade",
        "vps-pfad",
        "api.qrserver.com",
        "qrserver.com",
        "public server",
        "public link",
        "oeffentlicher server",
        "oeffentlicher link",
        "offentlicher server",
        "offentlicher link",
    ];
    let hits = forbidden_markers
        .iter()
        .filter(|marker| lowered.contains(**marker))
        .copied()
        .collect::<Vec<_>>();
    if !hits.is_empty() {
        anyhow::bail!(
            "founder/owner outbound email failed communication review due to internal-language leakage: {}",
            hits.join(", ")
        );
    }
    Ok(())
}

fn send_email_message(
    root: &Path,
    conn: &Connection,
    db_path: &Path,
    request: &ChannelSendRequest,
    reviewed_context: Option<ReviewedFounderSendContext<'_>>,
) -> Result<Value> {
    let adapter = communication_adapters::email();
    let sender_email = request
        .sender_address
        .clone()
        .unwrap_or_else(|| email_address_from_account_key(&request.account_key));
    let account_config = load_account_config(conn, &request.account_key)?;
    let body_sha256 = sha256_hex(request.body.trim().as_bytes());
    let approval_key = reviewed_context
        .map(|context| context.approval_key)
        .unwrap_or("");
    // EGRESS-2: a crash between the provider send (adapter.send_cli) and the
    // accepted-mark strands a draft_pending_send row carrying a
    // send_attempt_started_at marker. Refuse to blind-resend that founder email
    // — the provider may already have delivered it — and require operator
    // verification instead of silently duplicating it. This runs BEFORE
    // record_outbound_pending_send, whose ON CONFLICT would overwrite the
    // marker on the stranded row.
    let stranded_message_key = pending_send_message_key(request, &body_sha256);
    if let Some(attempt_started_at) = stranded_outbound_send_attempt(conn, &stranded_message_key)? {
        anyhow::bail!(
            "refusing to re-send founder email {stranded_message_key}: a provider send was \
             initiated at {attempt_started_at} but never confirmed accepted (possible crash \
             between send and acknowledgement); verify delivery before resending"
        );
    }
    let pending_send = record_outbound_pending_send(conn, request, approval_key, &body_sha256)?;
    let pending_message_key = pending_send.message_key;
    if let Some(existing) = pending_send.existing_result {
        return Ok(json!({
            "ok": true,
            "channel": "email",
            "db_path": db_path,
            "message_key": pending_message_key,
            "status": existing
                .get("status")
                .and_then(Value::as_str)
                .unwrap_or("accepted"),
            "delivery_confirmed": existing
                .get("adapter_result")
                .or_else(|| existing.get("adapterResult"))
                .and_then(|value| value.get("delivery"))
                .and_then(|value| value.get("confirmed"))
                .and_then(Value::as_bool)
                .unwrap_or(false),
            "adapter_result": existing
                .get("adapter_result")
                .or_else(|| existing.get("adapterResult"))
                .cloned()
                .unwrap_or_else(|| json!({ "deduplicated": true })),
            "deduplicated": true,
        }));
    }
    // EGRESS-2: record that the provider call is about to happen BEFORE the
    // physical send, so a crash after send_cli but before the accepted-mark is
    // recoverable as "maybe sent" (stranded_outbound_send_attempt) rather than
    // an unconditional resend on the next attempt.
    mark_outbound_send_attempt_started(conn, &pending_message_key)?;
    let adapter_json = match adapter.send_cli(
        root,
        &communication_adapters::EmailSendCommandRequest {
            db_path,
            sender_email: &sender_email,
            provider: account_config
                .as_ref()
                .map(|config| config.provider.as_str()),
            profile_json: account_config.as_ref().map(|config| &config.profile_json),
            thread_key: &request.thread_key,
            to: &request.to,
            cc: &request.cc,
            sender_display: request.sender_display.as_deref(),
            subject: &request.subject,
            body: &request.body,
            attachments: &request.attachments,
        },
    ) {
        Ok(value) => value,
        Err(err) => {
            let _ = mark_outbound_send_failed(conn, &pending_message_key, &err.to_string());
            if let Some(context) = reviewed_context {
                let _ = enforce_reviewed_founder_send_failed_core_transition(
                    conn,
                    context.entity_id,
                    context.approval_key,
                    request,
                    &pending_message_key,
                    &err.to_string(),
                );
            }
            return Err(err);
        }
    };
    let status = adapter_json
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or("accepted");
    mark_outbound_send_accepted(conn, &pending_message_key, status, &adapter_json)?;
    if let Some(context) = reviewed_context {
        // The kernel must witness send SUCCESS too, symmetric to the failure
        // twin above, so a reviewed founder send reaches terminal Sent instead
        // of being stranded in non-terminal Sending. Best-effort: an
        // already-delivered mail must not be failed by a witness hiccup.
        let _ = enforce_reviewed_founder_send_succeeded_core_transition(
            conn,
            context.entity_id,
            context.approval_key,
            request,
            &pending_message_key,
        );
    }
    Ok(json!({
        "ok": true,
        "channel": "email",
        "db_path": db_path,
        "message_key": pending_message_key,
        "status": status,
        "delivery_confirmed": adapter_json
            .get("delivery")
            .and_then(|value| value.get("confirmed"))
            .and_then(Value::as_bool)
            .unwrap_or(false),
        "adapter_result": adapter_json,
    }))
}

#[derive(Debug, Clone, Copy)]
struct ReviewedFounderSendContext<'a> {
    entity_id: &'a str,
    approval_key: &'a str,
}

fn record_outbound_pending_send(
    conn: &Connection,
    request: &ChannelSendRequest,
    approval_key: &str,
    body_sha256: &str,
) -> Result<PendingSendReservation> {
    let observed_at = now_iso_string();
    let message_key = pending_send_message_key(request, body_sha256);
    if let Some(existing) = existing_durable_outbound_send_result(conn, &message_key)? {
        return Ok(PendingSendReservation {
            message_key,
            existing_result: Some(existing),
        });
    }
    let remote_id = format!("pending-send-{}", stable_digest(&message_key));
    let recipient_set_sha256 = founder_send_recipient_set_sha256(request);
    let sender_email = request
        .sender_address
        .clone()
        .unwrap_or_else(|| email_address_from_account_key(&request.account_key));
    let metadata_json = serde_json::to_string(&json!({
        "source": "ctox-send-durability",
        "pendingSend": true,
        "pending_send": true,
        "reviewedFounderSend": request.reviewed_founder_send,
        "attachments": request.attachments,
        "approval_key": approval_key,
        "body_sha256": body_sha256,
        "recipient_set_sha256": recipient_set_sha256,
        "phase": "phase1_body_durability",
    }))?;
    conn.execute(
        r#"
        INSERT INTO communication_messages (
            message_key, channel, account_key, thread_key, remote_id, direction, folder_hint,
            sender_display, sender_address, recipient_addresses_json, cc_addresses_json, bcc_addresses_json,
            subject, preview, body_text, body_html, raw_payload_ref, trust_level, status, seen,
            has_attachments, external_created_at, observed_at, metadata_json
        ) VALUES (
            ?1, 'email', ?2, ?3, ?4, 'outbound', 'outbox',
            ?5, ?6, ?7, ?8, '[]',
            ?9, ?10, ?11, '', ?12, 'high', 'draft_pending_send', 1,
            ?13, ?14, ?14, ?15
        )
        ON CONFLICT(message_key) DO UPDATE SET
            folder_hint='outbox',
            status='draft_pending_send',
            body_text=excluded.body_text,
            metadata_json=excluded.metadata_json,
            observed_at=excluded.observed_at
        WHERE communication_messages.status IN ('draft_pending_send', 'send_failed')
        "#,
        params![
            message_key,
            request.account_key,
            request.thread_key,
            remote_id,
            request.sender_display.as_deref().unwrap_or(""),
            sender_email,
            serde_json::to_string(&request.to)?,
            serde_json::to_string(&request.cc)?,
            request.subject,
            preview_text(&request.body, &request.subject),
            request.body,
            request.attachments.join("\n"),
            if request.attachments.is_empty() { 0 } else { 1 },
            observed_at,
            metadata_json,
        ],
    )?;
    if let Some(existing) = existing_durable_outbound_send_result(conn, &message_key)? {
        return Ok(PendingSendReservation {
            message_key,
            existing_result: Some(existing),
        });
    }
    Ok(PendingSendReservation {
        message_key,
        existing_result: None,
    })
}

#[derive(Debug)]
struct PendingSendReservation {
    message_key: String,
    existing_result: Option<Value>,
}

fn existing_durable_outbound_send_result(
    conn: &Connection,
    message_key: &str,
) -> Result<Option<Value>> {
    let existing = conn
        .query_row(
            r#"
            SELECT status, folder_hint, metadata_json
            FROM communication_messages
            WHERE message_key = ?1
              AND channel = 'email'
              AND direction = 'outbound'
            "#,
            params![message_key],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?;
    let Some((status, folder_hint, metadata_json)) = existing else {
        return Ok(None);
    };
    let metadata = serde_json::from_str::<Value>(&metadata_json).unwrap_or(Value::Null);
    if !is_durable_outbound_send_state(&status, &folder_hint, &metadata) {
        return Ok(None);
    }
    Ok(Some(json!({
        "status": status,
        "folder_hint": folder_hint,
        "adapter_result": metadata
            .get("adapterResult")
            .or_else(|| metadata.get("adapter_result"))
            .cloned()
            .unwrap_or_else(|| json!({})),
    })))
}

fn is_durable_outbound_send_state(status: &str, folder_hint: &str, metadata: &Value) -> bool {
    if !folder_hint.eq_ignore_ascii_case("sent") {
        return false;
    }
    if matches!(
        status,
        "draft_pending_send" | "send_failed" | "failed" | "cancelled"
    ) {
        return false;
    }
    !metadata
        .get("pendingSend")
        .or_else(|| metadata.get("pending_send"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
}

/// Whether an outbound row's provider send was already initiated but never
/// confirmed accepted — i.e. a not-yet-durable `draft_pending_send` row
/// carrying a `send_attempt_started_at` marker. Such a row is "maybe sent": a
/// process can crash after `adapter.send_cli` returns Ok but before
/// `mark_outbound_send_accepted` commits, and a blind resend would duplicate a
/// founder email. Returns the recorded attempt timestamp when stranded so the
/// caller can refuse the resend and require operator verification (EGRESS-2).
fn stranded_outbound_send_attempt(conn: &Connection, message_key: &str) -> Result<Option<String>> {
    let existing = conn
        .query_row(
            r#"
            SELECT status, folder_hint, metadata_json
            FROM communication_messages
            WHERE message_key = ?1
              AND channel = 'email'
              AND direction = 'outbound'
            "#,
            params![message_key],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?;
    let Some((status, folder_hint, metadata_json)) = existing else {
        return Ok(None);
    };
    let metadata = serde_json::from_str::<Value>(&metadata_json).unwrap_or(Value::Null);
    // A durable (accepted) row is handled by existing_durable_outbound_send_result.
    if is_durable_outbound_send_state(&status, &folder_hint, &metadata) {
        return Ok(None);
    }
    // Only a still-pending row is "maybe sent". A send_failed row means
    // adapter.send_cli returned Err (the provider rejected it = NOT delivered),
    // so it is safe to retry and must never be treated as stranded — and
    // mark_outbound_send_failed already clears the marker, so this is also
    // defense-in-depth against a lingering marker on a non-pending row.
    if status != "draft_pending_send" {
        return Ok(None);
    }
    Ok(metadata
        .get("send_attempt_started_at")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned))
}

/// Stamp `send_attempt_started_at` into the row metadata immediately before
/// `adapter.send_cli`, without disturbing the rest of the metadata, so a crash
/// in the send→accept window leaves a recoverable "maybe sent" marker that
/// `stranded_outbound_send_attempt` detects (EGRESS-2).
fn mark_outbound_send_attempt_started(conn: &Connection, message_key: &str) -> Result<()> {
    conn.execute(
        r#"
        UPDATE communication_messages
        SET metadata_json = json_set(metadata_json, '$.send_attempt_started_at', ?2)
        WHERE message_key = ?1
        "#,
        params![message_key, now_iso_string()],
    )?;
    Ok(())
}

fn mark_outbound_send_accepted(
    conn: &Connection,
    message_key: &str,
    status: &str,
    adapter_json: &Value,
) -> Result<()> {
    // Only transition a row that is still pending (or retrying after a prior
    // failure); never clobber a row that has already reached a terminal state.
    // A 0-row result means the send was already resolved, so this is an
    // idempotent no-op — NOT an error: the caller uses `?` on the success path,
    // and erroring there could trigger a re-send of an already-accepted message.
    let changed = conn.execute(
        r#"
        UPDATE communication_messages
        SET status = ?2,
            folder_hint = 'sent',
            metadata_json = json_set(
                json_remove(
                    json_set(metadata_json, '$.pendingSend', false),
                    '$.send_attempt_started_at'
                ),
                '$.adapterResult',
                json(?3)
            ),
            observed_at = ?4
        WHERE message_key = ?1
          AND status IN ('draft_pending_send', 'send_failed')
        "#,
        params![
            message_key,
            status,
            serde_json::to_string(adapter_json)?,
            now_iso_string()
        ],
    )?;
    if changed == 0 {
        eprintln!(
            "[ctox channels] mark_outbound_send_accepted: {message_key} was not in a pending state (already resolved); skipping idempotently"
        );
    }
    Ok(())
}

fn mark_outbound_send_failed(conn: &Connection, message_key: &str, error: &str) -> Result<()> {
    // Only mark a still-pending row as failed; a late or duplicate failure must
    // never clobber a row that has already been accepted (or cancelled). A
    // 0-row result is an idempotent no-op so a stray failure callback cannot
    // override a successful send.
    let changed = conn.execute(
        r#"
        UPDATE communication_messages
        SET status = 'send_failed',
            metadata_json = json_set(
                json_remove(
                    json_set(metadata_json, '$.pendingSend', false),
                    '$.send_attempt_started_at'
                ),
                '$.sendError',
                ?2
            ),
            observed_at = ?3
        WHERE message_key = ?1
          AND status IN ('draft_pending_send', 'send_failed')
        "#,
        params![message_key, error, now_iso_string()],
    )?;
    if changed == 0 {
        eprintln!(
            "[ctox channels] mark_outbound_send_failed: {message_key} was not in a pending state (already resolved); skipping idempotently"
        );
    }
    Ok(())
}

pub(crate) fn prepare_reviewed_founder_reply(
    root: &Path,
    inbound_message_key: &str,
) -> Result<FounderReplyAction> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let inbound = load_message_from_conn(&conn, inbound_message_key)?
        .with_context(|| format!("missing inbound communication message {inbound_message_key}"))?;
    anyhow::ensure!(
        inbound.channel == "email" && inbound.direction == "inbound",
        "reviewed founder reply requires an inbound email message"
    );
    let addressing = load_message_addressing_from_conn(&conn, inbound_message_key)?
        .with_context(|| format!("missing communication addressing for {inbound_message_key}"))?;
    let (to, cc) = derive_founder_reply_recipients(&inbound, &addressing);
    let attachments =
        prepare_founder_reply_attachments(root, &inbound.subject, &inbound.body_text)?;
    let request = resolve_outbound_subject(
        &conn,
        ChannelSendRequest {
            channel: "email".to_string(),
            account_key: inbound.account_key.clone(),
            thread_key: inbound.thread_key.clone(),
            body: String::new(),
            subject: format!("Re: {}", inbound.subject.trim()),
            to,
            cc,
            attachments,
            sender_display: None,
            sender_address: None,
            send_voice: false,
            reviewed_founder_send: true,
        },
    )?;
    Ok(FounderReplyAction {
        account_key: request.account_key,
        thread_key: request.thread_key,
        subject: request.subject,
        to: request.to,
        cc: request.cc,
        attachments: request.attachments,
    })
}

pub(crate) fn required_founder_reply_deliverables(
    root: &Path,
    inbound_message_key: &str,
) -> Result<Vec<String>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let inbound = load_message_from_conn(&conn, inbound_message_key)?
        .with_context(|| format!("missing inbound communication message {inbound_message_key}"))?;
    Ok(detect_required_founder_deliverables(
        &inbound.subject,
        &inbound.body_text,
    ))
}

pub(crate) fn ensure_founder_reply_deliverables_present(
    root: &Path,
    inbound_message_key: &str,
    body: &str,
    attachments: &[String],
) -> Result<()> {
    let required = required_founder_reply_deliverables(root, inbound_message_key)?;
    let missing = required
        .into_iter()
        .filter(|deliverable| !founder_reply_satisfies_deliverable(body, attachments, deliverable))
        .collect::<Vec<_>>();
    if !missing.is_empty() {
        anyhow::bail!(
            "founder reply is missing required deliverable(s): {}",
            missing.join(", ")
        );
    }
    Ok(())
}

pub(crate) fn record_founder_reply_review_approval(
    root: &Path,
    inbound_message_key: &str,
    body: &str,
    review_summary: &str,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let action = prepare_reviewed_founder_reply(root, inbound_message_key)?;
    let (action_digest, action_json, body_sha256) = founder_reply_review_digest(&action, body);
    let approval_key = format!("founder-review:{inbound_message_key}:{action_digest}");
    conn.execute(
        r#"
        INSERT INTO communication_founder_reply_reviews (
            approval_key, inbound_message_key, action_digest, action_json,
            body_sha256, reviewer, review_summary, approved_at, sent_at, send_result_json
        )
        VALUES (?1, ?2, ?3, ?4, ?5, 'external-review', ?6, ?7, NULL, '{}')
        ON CONFLICT(inbound_message_key, action_digest) DO UPDATE SET
            approval_key=excluded.approval_key,
            action_json=excluded.action_json,
            body_sha256=excluded.body_sha256,
            reviewer=excluded.reviewer,
            review_summary=excluded.review_summary,
            approved_at=excluded.approved_at,
            sent_at=NULL,
            send_result_json='{}'
        "#,
        params![
            approval_key,
            inbound_message_key,
            action_digest,
            action_json,
            body_sha256,
            review_summary,
            now_iso_string()
        ],
    )
    .context("failed to record founder reply review approval")?;
    record_harness_flow_event_lossy(
        root,
        RecordHarnessFlowEventRequest {
            event_kind: "review.approved",
            title: "Review approved",
            body_text: review_summary,
            message_key: Some(inbound_message_key),
            work_id: None,
            ticket_key: None,
            attempt_index: Some(1),
            metadata: json!({
                "approval_key": approval_key,
                "body_sha256": body_sha256,
                "action_digest": action_digest,
            }),
        },
    );
    Ok(())
}

fn founder_reply_review_digest(
    action: &FounderReplyAction,
    body: &str,
) -> (String, String, String) {
    let action_json = json!({
        "thread_key": &action.thread_key,
        "subject": &action.subject,
        "to": &action.to,
        "cc": &action.cc,
        "attachments": &action.attachments,
    })
    .to_string();
    let body_sha256 = format!("{:x}", Sha256::digest(body.trim().as_bytes()));
    let mut hasher = Sha256::new();
    hasher.update(action_json.as_bytes());
    hasher.update(b"\0");
    hasher.update(body_sha256.as_bytes());
    let action_digest = format!("{:x}", hasher.finalize());
    (action_digest, action_json, body_sha256)
}

fn founder_outbound_review_digest(
    action: &FounderOutboundAction,
    body: &str,
) -> (String, String, String) {
    let action_json = json!({
        "account_key": &action.account_key,
        "thread_key": &action.thread_key,
        "subject": &action.subject,
        "to": &action.to,
        "cc": &action.cc,
        "attachments": &action.attachments,
    })
    .to_string();
    let body_sha256 = format!("{:x}", Sha256::digest(body.trim().as_bytes()));
    let mut hasher = Sha256::new();
    hasher.update(action_json.as_bytes());
    hasher.update(b"\0");
    hasher.update(body_sha256.as_bytes());
    let action_digest = format!("{:x}", hasher.finalize());
    (action_digest, action_json, body_sha256)
}

fn external_chat_review_digest(
    action: &ExternalChatAction,
    body: &str,
) -> (String, String, String) {
    let review_kind = if action.channel.eq_ignore_ascii_case("email") {
        "reviewed_outbound_email"
    } else {
        "external_chat_quick_response"
    };
    let action_json = json!({
        "kind": review_kind,
        "channel": &action.channel,
        "account_key": &action.account_key,
        "thread_key": &action.thread_key,
        "subject": &action.subject,
        "to": &action.to,
        "cc": &action.cc,
        "attachments": &action.attachments,
    })
    .to_string();
    let body_sha256 = format!("{:x}", Sha256::digest(body.trim().as_bytes()));
    let mut hasher = Sha256::new();
    hasher.update(action_json.as_bytes());
    hasher.update(b"\0");
    hasher.update(body_sha256.as_bytes());
    let action_digest = format!("{:x}", hasher.finalize());
    (action_digest, action_json, body_sha256)
}

pub(crate) fn is_reviewed_external_chat_channel(channel: &str) -> bool {
    matches!(
        channel,
        "teams"
            | "jami"
            | "whatsapp"
            | "meeting"
            | "slack"
            | "discord"
            | "telegram"
            | "matrix"
            | "mattermost"
            | "zulip"
            | "google_chat"
    )
}

fn external_chat_action_from_send_request(request: &ChannelSendRequest) -> ExternalChatAction {
    ExternalChatAction {
        channel: request.channel.clone(),
        account_key: request.account_key.clone(),
        thread_key: request.thread_key.clone(),
        subject: request.subject.clone(),
        to: request.to.clone(),
        cc: request.cc.clone(),
        attachments: request.attachments.clone(),
    }
}

pub(crate) fn prepare_reviewed_external_chat_reply(
    root: &Path,
    inbound_message_key: &str,
) -> Result<Option<ExternalChatAction>> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let Some(inbound) = load_message_from_conn(&conn, inbound_message_key)? else {
        return Ok(None);
    };
    if !is_reviewed_external_chat_channel(&inbound.channel) || inbound.direction != "inbound" {
        return Ok(None);
    }
    let to = if inbound.channel == "jami" && !inbound.sender_address.trim().is_empty() {
        vec![inbound.sender_address.trim().to_string()]
    } else {
        Vec::new()
    };
    Ok(Some(ExternalChatAction {
        channel: inbound.channel,
        account_key: inbound.account_key,
        thread_key: inbound.thread_key,
        subject: inbound.subject,
        to,
        cc: Vec::new(),
        attachments: Vec::new(),
    }))
}

pub(crate) fn default_email_account_key(root: &Path) -> Result<String> {
    let db_path = resolve_db_path(root, None);
    bootstrap_channel_account(root, "email")?;
    let conn = open_channel_db(&db_path)?;
    resolve_account_key(&conn, "email", None)
}

pub(crate) fn terminal_founder_outbound_artifact_count(
    root: &Path,
    action: &FounderOutboundAction,
) -> Result<i64> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let to_json = serde_json::to_string(&action.to)?;
    let cc_json = serde_json::to_string(&action.cc)?;
    let attachments = action.attachments.join("\n");
    conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM communication_messages
        WHERE channel = 'email'
          AND direction = 'outbound'
          AND status IN ('accepted', 'sent', 'queued', 'queued_in_mailserver', 'queued_for_provider')
          AND lower(account_key) = lower(?1)
          AND thread_key = ?2
          AND subject = ?3
          AND recipient_addresses_json = ?4
          AND cc_addresses_json = ?5
          AND raw_payload_ref = ?6
        "#,
        params![
            action.account_key,
            action.thread_key,
            action.subject,
            to_json,
            cc_json,
            attachments
        ],
        |row| row.get(0),
    )
    .context("failed to count terminal founder outbound artifacts")
}

pub(crate) fn reviewed_send_result_has_durable_outbound_artifact(
    root: &Path,
    send_result: &Value,
) -> Result<bool> {
    let Some(message_key) = send_result
        .get("message_key")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty())
    else {
        return Ok(false);
    };
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let count: i64 = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM communication_messages
        WHERE message_key = ?1
          AND channel = 'email'
          AND direction = 'outbound'
          AND lower(COALESCE(folder_hint, '')) = 'sent'
          AND status NOT IN ('draft_pending_send', 'send_failed', 'failed', 'cancelled')
          AND COALESCE(
                json_extract(metadata_json, '$.pendingSend'),
                json_extract(metadata_json, '$.pending_send'),
                0
              ) = 0
        "#,
        params![message_key],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

pub(crate) fn record_founder_outbound_review_approval(
    root: &Path,
    anchor_message_key: &str,
    action: &FounderOutboundAction,
    body: &str,
    review_summary: &str,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let (action_digest, action_json, body_sha256) = founder_outbound_review_digest(action, body);
    let approval_key = format!("founder-outbound-review:{anchor_message_key}:{action_digest}");
    conn.execute(
        r#"
        INSERT INTO communication_founder_reply_reviews (
            approval_key, inbound_message_key, action_digest, action_json,
            body_sha256, reviewer, review_summary, approved_at, sent_at, send_result_json
        )
        VALUES (?1, ?2, ?3, ?4, ?5, 'external-review', ?6, ?7, NULL, '{}')
        ON CONFLICT(inbound_message_key, action_digest) DO UPDATE SET
            approval_key=excluded.approval_key,
            action_json=excluded.action_json,
            body_sha256=excluded.body_sha256,
            reviewer=excluded.reviewer,
            review_summary=excluded.review_summary,
            approved_at=excluded.approved_at,
            sent_at=NULL,
            send_result_json='{}'
        "#,
        params![
            approval_key,
            anchor_message_key,
            action_digest,
            action_json,
            body_sha256,
            review_summary,
            now_iso_string()
        ],
    )
    .context("failed to record founder outbound review approval")?;
    record_harness_flow_event_lossy(
        root,
        RecordHarnessFlowEventRequest {
            event_kind: "review.approved",
            title: "Review approved",
            body_text: review_summary,
            message_key: Some(anchor_message_key),
            work_id: None,
            ticket_key: None,
            attempt_index: Some(1),
            metadata: json!({
                "approval_key": approval_key,
                "body_sha256": body_sha256,
                "action_digest": action_digest,
                "outbound": true,
            }),
        },
    );
    Ok(())
}

pub(crate) fn record_external_chat_review_approval(
    root: &Path,
    anchor_message_key: &str,
    action: &ExternalChatAction,
    body: &str,
    review_summary: &str,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let (action_digest, action_json, body_sha256) = external_chat_review_digest(action, body);
    let approval_prefix = if action.channel.eq_ignore_ascii_case("email") {
        "communication-email-review"
    } else {
        "external-chat-review"
    };
    let approval_key = format!("{approval_prefix}:{anchor_message_key}:{action_digest}");
    conn.execute(
        r#"
        INSERT INTO communication_founder_reply_reviews (
            approval_key, inbound_message_key, action_digest, action_json,
            body_sha256, reviewer, review_summary, approved_at, sent_at, send_result_json
        )
        VALUES (?1, ?2, ?3, ?4, ?5, 'external-review', ?6, ?7, NULL, '{}')
        ON CONFLICT(inbound_message_key, action_digest) DO UPDATE SET
            approval_key=excluded.approval_key,
            action_json=excluded.action_json,
            body_sha256=excluded.body_sha256,
            reviewer=excluded.reviewer,
            review_summary=excluded.review_summary,
            approved_at=excluded.approved_at,
            sent_at=NULL,
            send_result_json='{}'
        "#,
        params![
            approval_key,
            anchor_message_key,
            action_digest,
            action_json,
            body_sha256,
            review_summary,
            now_iso_string()
        ],
    )
    .context("failed to record communication review approval")?;
    let email_review = action.channel.eq_ignore_ascii_case("email");
    record_harness_flow_event_lossy(
        root,
        RecordHarnessFlowEventRequest {
            event_kind: "review.approved",
            title: if email_review {
                "Email communication review approved"
            } else {
                "External chat review approved"
            },
            body_text: review_summary,
            message_key: Some(anchor_message_key),
            work_id: None,
            ticket_key: None,
            attempt_index: Some(1),
            metadata: json!({
                "approval_key": approval_key,
                "body_sha256": body_sha256,
                "action_digest": action_digest,
                "channel": &action.channel,
                "communication_review": true,
                "email": email_review,
                "external_chat": !email_review,
            }),
        },
    );
    Ok(())
}

/// Persist a structured "no-send" verdict for an inbound message. The
/// terminal NO-SEND disposition is identified by a synthetic
/// `terminal-no-send:<inbound>` digest; it does not reference any
/// outbound action because the whole point of the verdict is that no
/// reply is going to be drafted. Re-recording is idempotent: the
/// underlying UNIQUE(inbound_message_key, action_digest) constraint
/// upserts on conflict.
pub fn record_terminal_no_send_verdict(
    root: &Path,
    inbound_message_key: &str,
    reviewer: &str,
    review_summary: &str,
) -> Result<()> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let action_digest = format!(
        "{:x}",
        Sha256::digest(format!("terminal-no-send:{inbound_message_key}").as_bytes())
    );
    let approval_key = format!("founder-no-send:{inbound_message_key}:{action_digest}");
    let action_json = json!({
        "kind": "terminal_no_send",
        "inbound_message_key": inbound_message_key,
    })
    .to_string();
    let body_sha256 = format!("{:x}", Sha256::digest(b""));
    conn.execute(
        r#"
        INSERT INTO communication_founder_reply_reviews (
            approval_key, inbound_message_key, action_digest, action_json,
            body_sha256, reviewer, review_summary, approved_at, sent_at,
            send_result_json, terminal_no_send
        )
        VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, NULL, '{}', 1)
        ON CONFLICT(inbound_message_key, action_digest) DO UPDATE SET
            approval_key=excluded.approval_key,
            action_json=excluded.action_json,
            reviewer=excluded.reviewer,
            review_summary=excluded.review_summary,
            approved_at=excluded.approved_at,
            terminal_no_send=1
        "#,
        params![
            approval_key,
            inbound_message_key,
            action_digest,
            action_json,
            body_sha256,
            reviewer,
            review_summary,
            now_iso_string()
        ],
    )
    .context("failed to record terminal NO-SEND verdict")?;
    record_harness_flow_event_lossy(
        root,
        RecordHarnessFlowEventRequest {
            event_kind: "review.no_send",
            title: "Review verdict: no-send",
            body_text: review_summary,
            message_key: Some(inbound_message_key),
            work_id: None,
            ticket_key: None,
            attempt_index: Some(1),
            metadata: json!({
                "approval_key": approval_key,
                "terminal_no_send": true,
            }),
        },
    );
    Ok(())
}

/// Whether a structured terminal NO-SEND verdict has been recorded for
/// the inbound message. Callers (notably the rework-spawn gate) must
/// query this BEFORE creating new founder-communication rework, so a
/// later auto-classifier cannot overwrite the original NO-SEND review.
pub fn inbound_message_has_terminal_no_send(
    root: &Path,
    inbound_message_key: &str,
) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let exists: i64 = conn.query_row(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM communication_founder_reply_reviews
            WHERE inbound_message_key = ?1
              AND terminal_no_send = 1
            LIMIT 1
        )
        "#,
        params![inbound_message_key],
        |row| row.get(0),
    )?;
    Ok(exists != 0)
}

/// Whether an inbound message is structurally non-actionable (i.e. an
/// auto-submitted/out-of-office reply per RFC 3834). The check looks
/// only at the metadata JSON written by the inbound parser; subject
/// and body text are not inspected here.
pub fn inbound_message_is_auto_submitted(root: &Path, inbound_message_key: &str) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let row: Option<String> = conn
        .query_row(
            "SELECT metadata_json FROM communication_messages WHERE message_key = ?1",
            params![inbound_message_key],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .context("failed to load inbound metadata for auto-submitted check")?;
    let Some(raw) = row else {
        return Ok(false);
    };
    let metadata: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
    Ok(metadata_marks_auto_submitted(&metadata))
}

fn require_unconsumed_founder_reply_review(
    conn: &Connection,
    inbound_message_key: &str,
    action: &FounderReplyAction,
    body: &str,
) -> Result<String> {
    let (action_digest, _, _) = founder_reply_review_digest(action, body);
    let approval_key = conn
        .query_row(
            r#"
            SELECT approval_key
            FROM communication_founder_reply_reviews
            WHERE inbound_message_key = ?1
              AND action_digest = ?2
              AND sent_at IS NULL
            LIMIT 1
            "#,
            params![inbound_message_key, action_digest],
            |row| row.get::<_, String>(0),
        )
        .optional()
        .context("failed to load founder reply review approval")?;
    approval_key.with_context(|| {
        "reviewed founder reply has no matching unconsumed review approval for the exact body, recipients, cc, subject, and attachments"
            .to_string()
    })
}

fn require_any_unconsumed_founder_outbound_review(
    conn: &Connection,
    action: &FounderOutboundAction,
    body: &str,
) -> Result<(String, String)> {
    let (action_digest, _, _) = founder_outbound_review_digest(action, body);
    let approval = conn
        .query_row(
            r#"
            SELECT approval_key, inbound_message_key
            FROM communication_founder_reply_reviews
            WHERE action_digest = ?1
              AND sent_at IS NULL
              AND terminal_no_send = 0
            ORDER BY approved_at DESC
            LIMIT 1
            "#,
            params![action_digest],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .context("failed to load founder outbound review approval")?;
    approval.with_context(|| {
        "reviewed founder outbound has no matching unconsumed review approval for the exact body, recipients, cc, subject, and attachments. Run completion review first, then send exactly the approved body with the same recipients and subject."
            .to_string()
    })
}

fn require_any_unconsumed_external_chat_review(
    conn: &Connection,
    action: &ExternalChatAction,
    body: &str,
) -> Result<(String, String)> {
    let (action_digest, _, _) = external_chat_review_digest(action, body);
    let approval = conn
        .query_row(
            r#"
            SELECT approval_key, inbound_message_key
            FROM communication_founder_reply_reviews
            WHERE action_digest = ?1
              AND sent_at IS NULL
              AND terminal_no_send = 0
            ORDER BY approved_at DESC
            LIMIT 1
            "#,
            params![action_digest],
            |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()
        .context("failed to load communication review approval")?;
    approval.with_context(|| {
        "reviewed outbound communication has no matching unconsumed review approval for the exact body, channel, thread, recipients, subject, and attachments. Run completion review first; after approval the Harness sends exactly the approved body."
            .to_string()
    })
}

fn mark_founder_reply_review_sent(
    conn: &Connection,
    approval_key: &str,
    send_result: &Value,
) -> Result<()> {
    conn.execute(
        r#"
        UPDATE communication_founder_reply_reviews
        SET sent_at = ?2,
            send_result_json = ?3
        WHERE approval_key = ?1
          AND sent_at IS NULL
        "#,
        params![approval_key, now_iso_string(), send_result.to_string()],
    )
    .context("failed to mark founder reply review as sent")?;
    Ok(())
}

fn founder_reply_sent_after_review(conn: &Connection, inbound_message_key: &str) -> Result<bool> {
    let count: i64 = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM communication_founder_reply_reviews
        WHERE inbound_message_key = ?1
          AND sent_at IS NOT NULL
          AND COALESCE(json_extract(send_result_json, '$.synthetic'), 0) != 1
          AND COALESCE(json_extract(send_result_json, '$.status'), '') != 'no-send-recorded'
        "#,
        params![inbound_message_key],
        |row| row.get(0),
    )?;
    Ok(count > 0)
}

fn protected_founder_inbound_message(
    root: &Path,
    conn: &Connection,
    message_key: &str,
) -> Result<bool> {
    let Some((channel, direction, sender_address)) = conn
        .query_row(
            r#"
            SELECT channel, direction, sender_address
            FROM communication_messages
            WHERE message_key = ?1
            LIMIT 1
            "#,
            params![message_key],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                ))
            },
        )
        .optional()?
    else {
        return Ok(false);
    };
    if channel != "email" || direction != "inbound" {
        return Ok(false);
    }
    let settings = runtime_settings_with_owner_profiles(
        root,
        communication_gateway::CommunicationAdapterKind::Email,
    );
    let policy = classify_email_sender(&settings, &sender_address);
    Ok(matches!(
        policy.role.as_str(),
        "owner" | "founder" | "admin"
    ))
}

fn message_metadata_marks_auto_submitted(conn: &Connection, message_key: &str) -> Result<bool> {
    let raw: Option<String> = conn
        .query_row(
            "SELECT metadata_json FROM communication_messages WHERE message_key = ?1",
            params![message_key],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let Some(raw) = raw else {
        return Ok(false);
    };
    let metadata: Value = serde_json::from_str(&raw).unwrap_or(Value::Null);
    Ok(metadata_marks_auto_submitted(&metadata))
}

fn message_has_terminal_no_send_in_conn(conn: &Connection, message_key: &str) -> Result<bool> {
    let exists: i64 = conn.query_row(
        r#"
        SELECT EXISTS(
            SELECT 1
            FROM communication_founder_reply_reviews
            WHERE inbound_message_key = ?1
              AND terminal_no_send = 1
            LIMIT 1
        )
        "#,
        params![message_key],
        |row| row.get(0),
    )?;
    Ok(exists != 0)
}

fn guard_founder_handled_ack(
    root: &Path,
    conn: &Connection,
    message_keys: &[String],
    status: &str,
) -> Result<()> {
    if status != "handled" {
        return Ok(());
    }
    for message_key in message_keys {
        if !protected_founder_inbound_message(root, conn, message_key)? {
            continue;
        }
        if founder_reply_sent_after_review(conn, message_key)? {
            continue;
        }
        // Bug #1: an auto-submitted (RFC 3834) founder/owner/admin
        // mail does not require a reviewed reply. The structured
        // header marker is checked at ingestion time and persisted
        // into metadata_json; we only consult the structured field
        // here, never subject/body strings.
        if message_metadata_marks_auto_submitted(conn, message_key)? {
            continue;
        }
        // Bug #3: an explicit terminal NO-SEND verdict closes the
        // inbound without a reply.
        if message_has_terminal_no_send_in_conn(conn, message_key)? {
            continue;
        }
        anyhow::bail!(
            "cannot mark founder/owner/admin inbound mail as handled before an exact reviewed reply was accepted by the email adapter: {}",
            message_key
        );
    }
    Ok(())
}

pub fn send_reviewed_founder_reply(
    root: &Path,
    inbound_message_key: &str,
    body: &str,
) -> Result<Value> {
    let _send_guard = acquire_reviewed_founder_send_lock()?;
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let inbound = load_message_from_conn(&conn, inbound_message_key)?
        .with_context(|| format!("missing inbound communication message {inbound_message_key}"))?;
    let action = prepare_reviewed_founder_reply(root, inbound_message_key)?;
    let request = resolve_outbound_subject(
        &conn,
        ChannelSendRequest {
            channel: "email".to_string(),
            account_key: inbound.account_key.clone(),
            thread_key: action.thread_key.clone(),
            body: body.trim().to_string(),
            subject: action.subject.clone(),
            to: action.to.clone(),
            cc: action.cc.clone(),
            attachments: action.attachments.clone(),
            sender_display: None,
            sender_address: None,
            send_voice: false,
            reviewed_founder_send: true,
        },
    )?;
    let settings = runtime_settings_with_owner_profiles(
        root,
        communication_gateway::CommunicationAdapterKind::Email,
    );
    let protected = protected_recipient_policies(&settings, &request);
    anyhow::ensure!(
        !protected.is_empty(),
        "reviewed founder reply requires founder/owner/admin recipient"
    );
    let approval_key = require_unconsumed_founder_reply_review(
        &conn,
        inbound_message_key,
        &action,
        &request.body,
    )?;
    ensure_founder_outbound_body_clean(&request)?;
    ensure_founder_reply_deliverables_present(
        root,
        inbound_message_key,
        &request.body,
        &request.attachments,
    )?;
    let entity_id = format!("founder-reply:{inbound_message_key}");
    enforce_reviewed_founder_send_core_transition(&conn, &entity_id, &approval_key, &request)?;
    let send_result = send_email_message(
        root,
        &conn,
        &db_path,
        &request,
        Some(ReviewedFounderSendContext {
            entity_id: &entity_id,
            approval_key: &approval_key,
        }),
    )?;
    mark_founder_reply_review_sent(&conn, &approval_key, &send_result)?;
    Ok(send_result)
}

fn send_reviewed_founder_outbound_request(
    root: &Path,
    conn: &Connection,
    db_path: &Path,
    request: &ChannelSendRequest,
) -> Result<Value> {
    let _send_guard = acquire_reviewed_founder_send_lock()?;
    let action = FounderOutboundAction {
        account_key: request.account_key.clone(),
        thread_key: request.thread_key.clone(),
        subject: request.subject.clone(),
        to: request.to.clone(),
        cc: request.cc.clone(),
        attachments: request.attachments.clone(),
    };
    let (approval_key, anchor_message_key) =
        require_any_unconsumed_founder_outbound_review(conn, &action, &request.body)?;
    ensure_founder_outbound_body_clean(request)?;
    let entity_id = format!("founder-outbound:{anchor_message_key}");
    enforce_reviewed_founder_send_core_transition(conn, &entity_id, &approval_key, request)?;
    let send_result = send_email_message(
        root,
        conn,
        db_path,
        request,
        Some(ReviewedFounderSendContext {
            entity_id: &entity_id,
            approval_key: &approval_key,
        }),
    )?;
    mark_founder_reply_review_sent(conn, &approval_key, &send_result)?;
    Ok(send_result)
}

pub(crate) fn send_reviewed_founder_outbound_action(
    root: &Path,
    action: &FounderOutboundAction,
    body: &str,
) -> Result<Value> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let request = resolve_outbound_subject(
        &conn,
        ChannelSendRequest {
            channel: "email".to_string(),
            account_key: action.account_key.clone(),
            thread_key: action.thread_key.clone(),
            body: body.trim().to_string(),
            subject: action.subject.clone(),
            to: action.to.clone(),
            cc: action.cc.clone(),
            attachments: action.attachments.clone(),
            sender_display: None,
            sender_address: None,
            send_voice: false,
            reviewed_founder_send: true,
        },
    )?;
    send_reviewed_founder_outbound_request(root, &conn, &db_path, &request)
}

pub(crate) fn send_reviewed_external_chat_action(
    root: &Path,
    action: &ExternalChatAction,
    body: &str,
) -> Result<Value> {
    let db_path = resolve_db_path(root, None);
    send_message(
        root,
        &db_path,
        ChannelSendRequest {
            channel: action.channel.clone(),
            account_key: action.account_key.clone(),
            thread_key: action.thread_key.clone(),
            body: body.trim().to_string(),
            subject: action.subject.clone(),
            to: action.to.clone(),
            cc: action.cc.clone(),
            attachments: action.attachments.clone(),
            sender_display: None,
            sender_address: None,
            send_voice: false,
            reviewed_founder_send: true,
        },
    )
}

/// Deterministic policy escalation for a founder/owner inbound email whose
/// finite completion-review budget is exhausted: record a policy-authored
/// approval for the exact escalation body and send it through the same gated
/// reviewed-send sequence as `send_reviewed_founder_reply` (send lock,
/// protected-recipient check, exact-digest approval match, body-clean gate,
/// core transition, durable send artifact). The deliverables-presence gate is
/// intentionally not applied: the escalation exists precisely because the
/// requested deliverable could not be produced, and it must still reach the
/// founder instead of the thread ending silently.
pub(crate) fn record_and_send_founder_escalation_reply(
    root: &Path,
    inbound_message_key: &str,
    body: &str,
    review_summary: &str,
) -> Result<Value> {
    let _send_guard = acquire_reviewed_founder_send_lock()?;
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let inbound = load_message_from_conn(&conn, inbound_message_key)?
        .with_context(|| format!("missing inbound communication message {inbound_message_key}"))?;
    anyhow::ensure!(
        inbound.channel == "email" && inbound.direction == "inbound",
        "founder escalation reply requires an inbound email message"
    );
    let addressing = load_message_addressing_from_conn(&conn, inbound_message_key)?
        .with_context(|| format!("missing communication addressing for {inbound_message_key}"))?;
    let (to, cc) = derive_founder_reply_recipients(&inbound, &addressing);
    let request = resolve_outbound_subject(
        &conn,
        ChannelSendRequest {
            channel: "email".to_string(),
            account_key: inbound.account_key.clone(),
            thread_key: inbound.thread_key.clone(),
            body: body.trim().to_string(),
            subject: format!("Re: {}", inbound.subject.trim()),
            to,
            cc,
            attachments: Vec::new(),
            sender_display: None,
            sender_address: None,
            send_voice: false,
            reviewed_founder_send: true,
        },
    )?;
    let settings = runtime_settings_with_owner_profiles(
        root,
        communication_gateway::CommunicationAdapterKind::Email,
    );
    let protected = protected_recipient_policies(&settings, &request);
    anyhow::ensure!(
        !protected.is_empty(),
        "founder escalation reply requires founder/owner/admin recipient"
    );
    let action = FounderReplyAction {
        account_key: request.account_key.clone(),
        thread_key: request.thread_key.clone(),
        subject: request.subject.clone(),
        to: request.to.clone(),
        cc: request.cc.clone(),
        attachments: Vec::new(),
    };
    let (action_digest, action_json, body_sha256) =
        founder_reply_review_digest(&action, &request.body);
    let approval_key = format!("founder-escalation:{inbound_message_key}:{action_digest}");
    // A consumed escalation approval (sent_at set) must stay consumed: the
    // conflict arm deliberately does not reset sent_at, so a retry after a
    // successful send fails the unconsumed-approval lookup below instead of
    // double-sending the notice.
    conn.execute(
        r#"
        INSERT INTO communication_founder_reply_reviews (
            approval_key, inbound_message_key, action_digest, action_json,
            body_sha256, reviewer, review_summary, approved_at, sent_at, send_result_json
        )
        VALUES (?1, ?2, ?3, ?4, ?5, 'policy-escalation', ?6, ?7, NULL, '{}')
        ON CONFLICT(inbound_message_key, action_digest) DO UPDATE SET
            review_summary=excluded.review_summary
        "#,
        params![
            approval_key,
            inbound_message_key,
            action_digest,
            action_json,
            body_sha256,
            review_summary,
            now_iso_string()
        ],
    )
    .context("failed to record founder escalation approval")?;
    let approval_key = require_unconsumed_founder_reply_review(
        &conn,
        inbound_message_key,
        &action,
        &request.body,
    )?;
    ensure_founder_outbound_body_clean(&request)?;
    let entity_id = format!("founder-reply:{inbound_message_key}");
    enforce_reviewed_founder_send_core_transition(&conn, &entity_id, &approval_key, &request)?;
    let send_result = send_email_message(
        root,
        &conn,
        &db_path,
        &request,
        Some(ReviewedFounderSendContext {
            entity_id: &entity_id,
            approval_key: &approval_key,
        }),
    )?;
    mark_founder_reply_review_sent(&conn, &approval_key, &send_result)?;
    record_harness_flow_event_lossy(
        root,
        RecordHarnessFlowEventRequest {
            event_kind: "communication.escalated",
            title: "Founder communication escalated after exhausted rework budget",
            body_text: review_summary,
            message_key: Some(inbound_message_key),
            work_id: None,
            ticket_key: None,
            attempt_index: Some(1),
            metadata: json!({
                "approval_key": approval_key,
                "body_sha256": body_sha256,
                "action_digest": action_digest,
                "escalation": true,
            }),
        },
    );
    Ok(send_result)
}

/// Chat-channel counterpart of `record_and_send_founder_escalation_reply`:
/// record a policy-authored approval for the exact escalation body against
/// the stalled inbound chat message and deliver it through the reviewed
/// external-chat send path (exact-digest approval match plus core send
/// transition inside `send_message`).
pub(crate) fn record_and_send_external_chat_escalation_reply(
    root: &Path,
    inbound_message_key: &str,
    body: &str,
    review_summary: &str,
) -> Result<Value> {
    let action =
        prepare_reviewed_external_chat_reply(root, inbound_message_key)?.with_context(|| {
            format!("inbound {inbound_message_key} is not a reviewed external chat message")
        })?;
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let trimmed_body = body.trim();
    let (action_digest, action_json, body_sha256) =
        external_chat_review_digest(&action, trimmed_body);
    let approval_key = format!("external-chat-escalation:{inbound_message_key}:{action_digest}");
    conn.execute(
        r#"
        INSERT INTO communication_founder_reply_reviews (
            approval_key, inbound_message_key, action_digest, action_json,
            body_sha256, reviewer, review_summary, approved_at, sent_at, send_result_json
        )
        VALUES (?1, ?2, ?3, ?4, ?5, 'policy-escalation', ?6, ?7, NULL, '{}')
        ON CONFLICT(inbound_message_key, action_digest) DO UPDATE SET
            review_summary=excluded.review_summary
        "#,
        params![
            approval_key,
            inbound_message_key,
            action_digest,
            action_json,
            body_sha256,
            review_summary,
            now_iso_string()
        ],
    )
    .context("failed to record external chat escalation approval")?;
    record_harness_flow_event_lossy(
        root,
        RecordHarnessFlowEventRequest {
            event_kind: "communication.escalated",
            title: "External chat communication escalated after exhausted rework budget",
            body_text: review_summary,
            message_key: Some(inbound_message_key),
            work_id: None,
            ticket_key: None,
            attempt_index: Some(1),
            metadata: json!({
                "approval_key": approval_key,
                "body_sha256": body_sha256,
                "action_digest": action_digest,
                "channel": &action.channel,
                "escalation": true,
            }),
        },
    );
    drop(conn);
    send_reviewed_external_chat_action(root, &action, trimmed_body)
}

fn send_reviewed_email_communication_request(
    root: &Path,
    conn: &Connection,
    db_path: &Path,
    request: &ChannelSendRequest,
    approval_key: &str,
) -> Result<Value> {
    ensure_founder_outbound_body_clean(request)?;
    let entity_id = format!(
        "reviewed-email:{}:{}",
        request.thread_key,
        stable_digest(approval_key)
    );
    enforce_reviewed_founder_send_core_transition(conn, &entity_id, approval_key, request)?;
    let send_result = send_email_message(
        root,
        conn,
        db_path,
        request,
        Some(ReviewedFounderSendContext {
            entity_id: &entity_id,
            approval_key,
        }),
    )?;
    mark_founder_reply_review_sent(conn, approval_key, &send_result)?;
    Ok(send_result)
}

fn enforce_reviewed_communication_send_core_transition_if_approved(
    conn: &Connection,
    request: &ChannelSendRequest,
    approval: Option<&(String, String)>,
) -> Result<Option<String>> {
    let Some((approval_key, anchor_message_key)) = approval else {
        return Ok(None);
    };
    let entity_id = format!(
        "reviewed-communication:{}:{}",
        request.channel,
        stable_digest(&format!("{anchor_message_key}:{approval_key}"))
    );
    enforce_reviewed_founder_send_core_transition(conn, &entity_id, approval_key, request)?;
    Ok(Some(entity_id))
}

fn enforce_reviewed_founder_send_core_transition(
    conn: &Connection,
    entity_id: &str,
    approval_key: &str,
    request: &ChannelSendRequest,
) -> Result<()> {
    let mut metadata = BTreeMap::new();
    metadata.insert("protected_party".to_string(), "founder".to_string());
    metadata.insert("thread_key".to_string(), request.thread_key.clone());
    metadata.insert("subject".to_string(), request.subject.clone());
    metadata.insert("account_key".to_string(), request.account_key.clone());

    enforce_core_transition(
        conn,
        &CoreTransitionRequest {
            entity_type: CoreEntityType::FounderCommunication,
            entity_id: entity_id.to_string(),
            lane: RuntimeLane::P0FounderCommunication,
            from_state: CoreState::Approved,
            to_state: CoreState::Sending,
            event: CoreEvent::Send,
            actor: "ctox-reviewed-founder-send".to_string(),
            // EGRESS-3: approved hashes from the durable review record, outgoing
            // from the live request — the kernel require_reviewed_outbound gate is
            // now load-bearing instead of comparing the request against itself.
            evidence: reviewed_outbound_evidence(conn, approval_key, request),
            metadata,
        },
    )?;
    Ok(())
}

fn acquire_reviewed_founder_send_lock() -> Result<MutexGuard<'static, ()>> {
    REVIEWED_FOUNDER_SEND_LOCK
        .get_or_init(|| Mutex::new(()))
        .lock()
        .map_err(|err| anyhow::anyhow!("reviewed founder send lock poisoned: {err}"))
}

/// Compute the deterministic `message_key` for a pending-send durable
/// outbound row. Stable for identical (account_key, thread_key, subject,
/// recipient set, body) tuples. This is the retry-binding key the
/// operator uses to resume after a provider failure (RFC 0001 §5.1).
fn pending_send_message_key(request: &ChannelSendRequest, body_sha256: &str) -> String {
    let recipient_set_sha256 = founder_send_recipient_set_sha256(request);
    let payload = format!(
        "{}|{}|{}|{}",
        request.account_key.trim(),
        request.thread_key.trim(),
        recipient_set_sha256,
        body_sha256
    );
    let digest = sha256_hex(payload.as_bytes());
    format!("{}::pending_send::{}", request.account_key.trim(), digest)
}

/// Flip a `draft_pending_send` row to `accepted` after a successful
/// provider call. The CAS on `status` is defensive: a concurrent failure-
/// path update would cause this to be a noop, which is safer than
/// silently overwriting.
fn update_pending_send_to_accepted(
    conn: &Connection,
    pending_message_key: &str,
    adapter_result: &Value,
) -> Result<()> {
    let prior_metadata = load_metadata_for_message(conn, pending_message_key)?;
    let mut metadata = prior_metadata
        .as_object()
        .cloned()
        .unwrap_or_else(serde_json::Map::new);
    metadata.insert("pending_send".to_string(), Value::Bool(false));
    metadata.insert(
        "transitioned_to".to_string(),
        Value::String("accepted".to_string()),
    );
    metadata.insert("adapter_result".to_string(), adapter_result.clone());
    let metadata_json = Value::Object(metadata).to_string();
    let now = now_iso_string();
    let updated = conn
        .execute(
            r#"
            UPDATE communication_messages
            SET status = 'accepted',
                metadata_json = ?2,
                observed_at = ?3
            WHERE message_key = ?1
              AND status = 'draft_pending_send'
            "#,
            params![pending_message_key, metadata_json, now],
        )
        .context("failed to mark outbound body as accepted")?;
    if updated == 0 {
        anyhow::bail!(
            "outbound durability row {} was not in draft_pending_send when accepted-update was attempted",
            pending_message_key
        );
    }
    Ok(())
}

/// Flip a `draft_pending_send` row to `send_failed` after a provider
/// failure. Body and recipients stay; the provider error is recorded in
/// `metadata_json` so the operator/retry path can read it.
fn update_pending_send_to_failed(
    conn: &Connection,
    pending_message_key: &str,
    error_text: &str,
) -> Result<()> {
    let prior_metadata = load_metadata_for_message(conn, pending_message_key)?;
    let mut metadata = prior_metadata
        .as_object()
        .cloned()
        .unwrap_or_else(serde_json::Map::new);
    metadata.insert("pending_send".to_string(), Value::Bool(false));
    metadata.insert(
        "transitioned_to".to_string(),
        Value::String("send_failed".to_string()),
    );
    metadata.insert(
        "provider_error".to_string(),
        Value::String(clip_error_text(error_text, 2000)),
    );
    let metadata_json = Value::Object(metadata).to_string();
    let now = now_iso_string();
    let updated = conn
        .execute(
            r#"
            UPDATE communication_messages
            SET status = 'send_failed',
                metadata_json = ?2,
                observed_at = ?3
            WHERE message_key = ?1
              AND status = 'draft_pending_send'
            "#,
            params![pending_message_key, metadata_json, now],
        )
        .context("failed to mark outbound body as send_failed")?;
    if updated == 0 {
        anyhow::bail!(
            "outbound durability row {} was not in draft_pending_send when send_failed-update was attempted",
            pending_message_key
        );
    }
    Ok(())
}

fn load_metadata_for_message(conn: &Connection, message_key: &str) -> Result<Value> {
    let raw: Option<String> = conn
        .query_row(
            "SELECT metadata_json FROM communication_messages WHERE message_key = ?1",
            params![message_key],
            |row| row.get(0),
        )
        .optional()
        .context("failed to load metadata_json for outbound durability row")?;
    match raw {
        Some(json) => Ok(serde_json::from_str::<Value>(&json).unwrap_or(Value::Null)),
        None => Ok(Value::Null),
    }
}

fn enforce_reviewed_founder_send_failed_core_transition(
    conn: &Connection,
    entity_id: &str,
    approval_key: &str,
    request: &ChannelSendRequest,
    pending_message_key: &str,
    provider_error: &str,
) -> Result<()> {
    emit_reviewed_founder_send_failed_transition(
        conn,
        entity_id,
        approval_key,
        request,
        pending_message_key,
        provider_error,
    )
}

/// Emit the `Sending -> SendFailed` core transition after a provider
/// failure. RFC 0001 Phase 1: the kernel must witness every founder-send
/// failure, and the durable pending body row is bound into metadata.
fn emit_reviewed_founder_send_failed_transition(
    conn: &Connection,
    entity_id: &str,
    approval_key: &str,
    request: &ChannelSendRequest,
    pending_message_key: &str,
    provider_error: &str,
) -> Result<()> {
    let mut metadata = BTreeMap::new();
    metadata.insert("protected_party".to_string(), "founder".to_string());
    metadata.insert("thread_key".to_string(), request.thread_key.clone());
    metadata.insert("subject".to_string(), request.subject.clone());
    metadata.insert("account_key".to_string(), request.account_key.clone());
    metadata.insert(
        "pending_message_key".to_string(),
        pending_message_key.to_string(),
    );
    metadata.insert(
        "provider_error".to_string(),
        clip_error_text(provider_error, 500),
    );

    enforce_core_transition(
        conn,
        &CoreTransitionRequest {
            entity_type: CoreEntityType::FounderCommunication,
            entity_id: entity_id.to_string(),
            lane: RuntimeLane::P0FounderCommunication,
            from_state: CoreState::Sending,
            to_state: CoreState::SendFailed,
            event: CoreEvent::Fail,
            actor: "ctox-reviewed-founder-send".to_string(),
            // EGRESS-3: approved hashes from the durable review record, outgoing
            // from the live request — the ->Sent confirmation and the symmetric
            // failure record carry the same load-bearing evidence as the Send gate.
            evidence: reviewed_outbound_evidence(conn, approval_key, request),
            metadata,
        },
    )?;
    Ok(())
}

fn enforce_reviewed_founder_send_succeeded_core_transition(
    conn: &Connection,
    entity_id: &str,
    approval_key: &str,
    request: &ChannelSendRequest,
    pending_message_key: &str,
) -> Result<()> {
    emit_reviewed_founder_send_succeeded_transition(
        conn,
        entity_id,
        approval_key,
        request,
        pending_message_key,
    )
}

/// Emit the `Sending -> Sent` core transition after a successful provider
/// send. RFC 0001 Phase 1: the kernel must witness every founder-send outcome,
/// success symmetric to the failure twin, so the entity reaches a terminal
/// Sent state instead of being stranded in non-terminal Sending.
fn emit_reviewed_founder_send_succeeded_transition(
    conn: &Connection,
    entity_id: &str,
    approval_key: &str,
    request: &ChannelSendRequest,
    pending_message_key: &str,
) -> Result<()> {
    let mut metadata = BTreeMap::new();
    metadata.insert("protected_party".to_string(), "founder".to_string());
    metadata.insert("thread_key".to_string(), request.thread_key.clone());
    metadata.insert("subject".to_string(), request.subject.clone());
    metadata.insert("account_key".to_string(), request.account_key.clone());
    metadata.insert(
        "pending_message_key".to_string(),
        pending_message_key.to_string(),
    );

    enforce_core_transition(
        conn,
        &CoreTransitionRequest {
            entity_type: CoreEntityType::FounderCommunication,
            entity_id: entity_id.to_string(),
            lane: RuntimeLane::P0FounderCommunication,
            from_state: CoreState::Sending,
            to_state: CoreState::Sent,
            event: CoreEvent::ConfirmDelivery,
            actor: "ctox-reviewed-founder-send".to_string(),
            // EGRESS-3: approved hashes from the durable review record, outgoing
            // from the live request — the ->Sent confirmation and the symmetric
            // failure record carry the same load-bearing evidence as the Send gate.
            evidence: reviewed_outbound_evidence(conn, approval_key, request),
            metadata,
        },
    )?;
    Ok(())
}

fn clip_error_text(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        let mut clipped: String = text.chars().take(max).collect();
        clipped.push_str("...");
        clipped
    }
}

fn founder_send_recipient_set_sha256(request: &ChannelSendRequest) -> String {
    recipient_set_sha256(
        &request.to,
        &request.cc,
        &request.subject,
        &request.attachments,
    )
}

/// EGRESS-3: the canonical recipient-set hash over (to, cc, subject,
/// attachments) with the exact normalization the founder-send gate uses — to/cc
/// trimmed + lowercased, attachments trimmed, all sorted, subject trimmed. Shared
/// by the live-request path (`founder_send_recipient_set_sha256`) and the
/// stored-approval path (`approved_outbound_evidence_hashes`) so the kernel
/// `require_reviewed_outbound` comparison is between two genuinely independent
/// values computed by IDENTICAL code — no normalization drift can false-reject a
/// legitimate send.
fn recipient_set_sha256(
    to: &[String],
    cc: &[String],
    subject: &str,
    attachments: &[String],
) -> String {
    let mut to = to
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .collect::<Vec<_>>();
    let mut cc = cc
        .iter()
        .map(|value| value.trim().to_ascii_lowercase())
        .collect::<Vec<_>>();
    let mut attachments = attachments
        .iter()
        .map(|value| value.trim().to_string())
        .collect::<Vec<_>>();
    to.sort();
    cc.sort();
    attachments.sort();
    let payload = json!({
        "to": to,
        "cc": cc,
        "subject": subject.trim(),
        "attachments": attachments,
    })
    .to_string();
    sha256_hex(payload.as_bytes())
}

/// EGRESS-3: load the APPROVED body + recipient-set hashes from the durable
/// review record by `approval_key`, so the kernel gate compares the stored
/// approval against the live request rather than the request against itself. The
/// recipient hash is derived from the stored `action_json` (to/cc/subject/
/// attachments — present in every review action shape) via the same
/// `recipient_set_sha256`. Returns `None` when no review row matches the key (or
/// its `action_json` cannot be parsed), so the caller can fall back to the
/// request-derived values and never NEWLY reject a previously-valid send.
fn approved_outbound_evidence_hashes(
    conn: &Connection,
    approval_key: &str,
) -> Option<(String, String)> {
    let (body_sha256, action_json): (String, String) = conn
        .query_row(
            "SELECT body_sha256, action_json FROM communication_founder_reply_reviews \
             WHERE approval_key = ?1 LIMIT 1",
            params![approval_key],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .optional()
        .ok()
        .flatten()?;
    let action: Value = serde_json::from_str(&action_json).ok()?;
    let string_list = |key: &str| -> Vec<String> {
        action
            .get(key)
            .and_then(Value::as_array)
            .map(|items| {
                items
                    .iter()
                    .filter_map(|item| item.as_str().map(str::to_string))
                    .collect()
            })
            .unwrap_or_default()
    };
    let to = string_list("to");
    let cc = string_list("cc");
    let attachments = string_list("attachments");
    let subject = action.get("subject").and_then(Value::as_str).unwrap_or("");
    let approved_recipient = recipient_set_sha256(&to, &cc, subject, &attachments);
    Some((body_sha256, approved_recipient))
}

/// EGRESS-3: build the `CoreEvidenceRefs` for a reviewed founder/owner-send
/// transition with the APPROVED hashes sourced from the durable review record
/// (independent of the live request) and the OUTGOING hashes from the request, so
/// every `require_reviewed_outbound`-gated transition (Approved->Sending,
/// Sending->Sent, plus the symmetric failure record) carries a genuinely
/// load-bearing comparison instead of a value-against-itself tautology. Falls
/// back to the request-derived values when the approval is not record-backed, so
/// a previously-valid send is never newly rejected.
fn reviewed_outbound_evidence(
    conn: &Connection,
    approval_key: &str,
    request: &ChannelSendRequest,
) -> CoreEvidenceRefs {
    let outgoing_body_sha256 = sha256_hex(request.body.trim().as_bytes());
    let outgoing_recipient_set_sha256 = founder_send_recipient_set_sha256(request);
    let (approved_body_sha256, approved_recipient_set_sha256) =
        approved_outbound_evidence_hashes(conn, approval_key).unwrap_or_else(|| {
            (
                outgoing_body_sha256.clone(),
                outgoing_recipient_set_sha256.clone(),
            )
        });
    CoreEvidenceRefs {
        review_audit_key: Some(approval_key.to_string()),
        approved_body_sha256: Some(approved_body_sha256),
        outgoing_body_sha256: Some(outgoing_body_sha256),
        approved_recipient_set_sha256: Some(approved_recipient_set_sha256),
        outgoing_recipient_set_sha256: Some(outgoing_recipient_set_sha256),
        ..CoreEvidenceRefs::default()
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn test_channel(
    root: &Path,
    db_path: &Path,
    channel: &str,
    account_key: Option<&str>,
) -> Result<Value> {
    macro_rules! test_chat_adapter {
        ($factory:ident, $channel:literal) => {{
            bootstrap_channel_account(root, $channel)?;
            let conn = open_channel_db(db_path)?;
            let adapter = communication_adapters::$factory();
            let resolved_account_key = resolve_account_key(&conn, $channel, account_key).ok();
            let account_config = resolved_account_key
                .as_deref()
                .and_then(|key| load_account_config(&conn, key).ok().flatten());
            let empty_profile = json!({});
            let adapter_json = adapter.test_cli(
                root,
                &communication_adapters::ChatTestCommandRequest {
                    db_path,
                    profile_json: account_config
                        .as_ref()
                        .map(|config| &config.profile_json)
                        .unwrap_or(&empty_profile),
                },
            )?;
            Ok(json!({
                "ok": adapter_json.get("ok").and_then(Value::as_bool).unwrap_or(false),
                "channel": $channel,
                "account_key": resolved_account_key,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }};
    }
    match channel {
        "tui" => Ok(json!({
            "ok": true,
            "channel": "tui",
            "status": "ready",
            "detail": "local TUI channel does not require external transport setup",
            "db_path": db_path,
        })),
        "email" => {
            bootstrap_channel_account(root, "email")?;
            let conn = open_channel_db(db_path)?;
            let resolved_account_key = resolve_account_key(&conn, "email", account_key)?;
            let account_config =
                load_account_config(&conn, &resolved_account_key)?.ok_or_else(|| {
                    anyhow::anyhow!("missing email account config for {}", resolved_account_key)
                })?;
            let adapter = communication_adapters::email();
            let resolved_email = email_address_from_account_key(&resolved_account_key);
            let adapter_json = adapter.test_cli(
                root,
                &communication_adapters::EmailTestCommandRequest {
                    db_path,
                    email_address: &resolved_email,
                    provider: &account_config.provider,
                    profile_json: &account_config.profile_json,
                },
            )?;
            Ok(json!({
                "ok": adapter_json.get("ok").and_then(Value::as_bool).unwrap_or(false),
                "channel": "email",
                "account_key": resolved_account_key,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        "jami" => {
            bootstrap_channel_account(root, "jami")?;
            let conn = open_channel_db(db_path)?;
            let resolved_account_key = resolve_account_key(&conn, "jami", account_key)?;
            let account_config =
                load_account_config(&conn, &resolved_account_key)?.ok_or_else(|| {
                    anyhow::anyhow!("missing jami account config for {}", resolved_account_key)
                })?;
            let adapter = communication_adapters::jami();
            let resolved_account_id = jami_address_from_account_key(&resolved_account_key);
            let adapter_json = adapter.test_cli(
                root,
                &communication_adapters::JamiTestCommandRequest {
                    db_path,
                    account_id: &resolved_account_id,
                    provider: &account_config.provider,
                    profile_json: &account_config.profile_json,
                },
            )?;
            Ok(json!({
                "ok": adapter_json.get("ok").and_then(Value::as_bool).unwrap_or(false),
                "channel": "jami",
                "account_key": resolved_account_key,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        "teams" => {
            bootstrap_channel_account(root, "teams")?;
            let conn = open_channel_db(db_path)?;
            let adapter = communication_adapters::teams();
            let resolved_account_key = resolve_account_key(&conn, "teams", account_key).ok();
            let account_config = resolved_account_key
                .as_deref()
                .and_then(|key| load_account_config(&conn, key).ok().flatten());
            let empty_profile = json!({});
            let resolved_tenant_id = account_config
                .as_ref()
                .and_then(|config| config.profile_json.get("tenantId"))
                .and_then(Value::as_str)
                .map(str::to_string)
                .unwrap_or_default();
            let adapter_json = adapter.test_cli(
                root,
                &communication_adapters::TeamsTestCommandRequest {
                    db_path,
                    tenant_id: &resolved_tenant_id,
                    profile_json: account_config
                        .as_ref()
                        .map(|config| &config.profile_json)
                        .unwrap_or(&empty_profile),
                },
            )?;
            Ok(json!({
                "ok": adapter_json.get("ok").and_then(Value::as_bool).unwrap_or(false),
                "channel": "teams",
                "account_key": resolved_account_key,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        "discord" => test_chat_adapter!(discord, "discord"),
        "google_chat" => test_chat_adapter!(google_chat, "google_chat"),
        "whatsapp" => {
            let conn = open_channel_db(db_path)?;
            let resolved_account_key = resolve_account_key(&conn, "whatsapp", account_key).ok();
            let adapter = communication_adapters::whatsapp();
            let adapter_json = adapter.test_cli(
                root,
                &communication_adapters::WhatsappTestCommandRequest {
                    db_path,
                    account_key: resolved_account_key.as_deref().or(account_key),
                },
            )?;
            Ok(json!({
                "ok": adapter_json.get("ok").and_then(Value::as_bool).unwrap_or(false),
                "channel": "whatsapp",
                "account_key": resolved_account_key,
                "db_path": db_path,
                "adapter_result": adapter_json,
            }))
        }
        "matrix" => test_chat_adapter!(matrix, "matrix"),
        "mattermost" => test_chat_adapter!(mattermost, "mattermost"),
        "slack" => test_chat_adapter!(slack, "slack"),
        "telegram" => test_chat_adapter!(telegram, "telegram"),
        "zulip" => test_chat_adapter!(zulip, "zulip"),
        other => anyhow::bail!("unsupported channel test target: {other}"),
    }
}

fn bootstrap_channel_account(root: &Path, channel: &str) -> Result<()> {
    match channel {
        "email" => {
            let settings = communication_gateway::runtime_settings_from_root(
                root,
                communication_gateway::CommunicationAdapterKind::Email,
            );
            if settings
                .get("CTO_EMAIL_ADDRESS")
                .map(|value| !value.trim().is_empty())
                .unwrap_or(false)
            {
                sync_prompt_identity(root, &settings)?;
            }
        }
        "jami" => {
            let mut settings = communication_gateway::runtime_settings_from_root(
                root,
                communication_gateway::CommunicationAdapterKind::Jami,
            );
            let configured_account_id = settings
                .get("CTO_JAMI_ACCOUNT_ID")
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .map(str::to_string);
            let configured_profile_name = settings
                .get("CTO_JAMI_PROFILE_NAME")
                .map(|value| value.trim())
                .filter(|value| !value.is_empty())
                .map(str::to_string);

            if configured_account_id.is_some() || configured_profile_name.is_some() {
                let resolved = communication_adapters::jami().resolve_account(
                    root,
                    &communication_adapters::JamiResolveAccountCommandRequest {
                        account_id: configured_account_id.as_deref(),
                        profile_name: configured_profile_name.as_deref(),
                    },
                )?;
                if resolved.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                    if let Some(account) =
                        resolved.get("resolvedAccount").and_then(Value::as_object)
                    {
                        if let Some(account_id) = account
                            .get("accountId")
                            .and_then(Value::as_str)
                            .filter(|v| !v.trim().is_empty())
                        {
                            settings
                                .insert("CTO_JAMI_ACCOUNT_ID".to_string(), account_id.to_string());
                        }
                        if let Some(profile_name) = account
                            .get("displayName")
                            .and_then(Value::as_str)
                            .filter(|v| !v.trim().is_empty())
                        {
                            settings.insert(
                                "CTO_JAMI_PROFILE_NAME".to_string(),
                                profile_name.to_string(),
                            );
                        }
                    }
                }
                sync_prompt_identity(root, &settings)?;
            }
        }
        "teams" => {}
        _ => {}
    }
    Ok(())
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
    // Local/configured chat transports do not require ad hoc recipients here:
    // tui is local, Teams and the bot-platform adapters can target configured
    // default destinations or destination markers in thread_key, meeting
    // broadcasts through the active Playwright session, and WhatsApp replies
    // target the chat encoded in thread_key. Email and Jami still need
    // explicit remote targets.
    let whatsapp_thread_reply = channel == "whatsapp" && thread_key.contains("::chat::");
    if !matches!(
        channel.as_str(),
        "tui"
            | "teams"
            | "meeting"
            | "slack"
            | "discord"
            | "telegram"
            | "matrix"
            | "mattermost"
            | "zulip"
            | "google_chat"
    ) && !whatsapp_thread_reply
        && to.is_empty()
    {
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
        attachments: collect_flag_values(args, "--attach-file"),
        sender_display: find_flag_value(args, "--sender-display").map(ToOwned::to_owned),
        sender_address: find_flag_value(args, "--sender-address").map(ToOwned::to_owned),
        send_voice: has_flag(args, "--send-voice"),
        reviewed_founder_send: has_flag(args, "--reviewed-founder-send")
            || has_flag(args, "--reviewed-communication-send"),
    })
}

fn validate_founder_outbound_email(
    settings: &BTreeMap<String, String>,
    request: &ChannelSendRequest,
) -> Result<()> {
    if request.channel != "email" {
        return Ok(());
    }
    let protected_recipients = request
        .to
        .iter()
        .chain(request.cc.iter())
        .map(|email| classify_email_sender(settings, email))
        .filter(|policy| matches!(policy.role.as_str(), "owner" | "founder" | "admin"))
        .collect::<Vec<_>>();
    if protected_recipients.is_empty() {
        anyhow::bail!(
            "direct outbound email is blocked without communication review. Draft the email for completion review first; after approval the Harness sends the exact approved body."
        );
    }
    let recipient_summary = protected_recipients
        .iter()
        .map(|policy| format!("{} ({})", policy.normalized_email, policy.role))
        .collect::<Vec<_>>()
        .join(", ");
    anyhow::ensure!(
        request.reviewed_founder_send,
        "direct outbound email to founder/owner/admin recipients is blocked without review: {}. Use a reviewed founder-send path.",
        recipient_summary
    );
    // Body-content guidance for mandantengerechte mail lives in
    // `owner-communication/SKILL.md`. CTOX core does not scrape the body for
    // internal vocabulary — the agent owns the wording, not the harness.
    anyhow::bail!(
        "generic channel send is disabled for founder/owner/admin outbound email: {}. Use the dedicated reviewed founder communication path instead.",
        recipient_summary
    );
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

fn thread_prefers_voice_reply(conn: &Connection, thread_key: &str) -> Result<bool> {
    let metadata_json = conn
        .query_row(
            r#"
            SELECT metadata_json
            FROM communication_messages
            WHERE thread_key = ?1
              AND direction = 'inbound'
            ORDER BY external_created_at DESC, observed_at DESC
            LIMIT 1
            "#,
            params![thread_key],
            |row| row.get::<_, String>(0),
        )
        .optional()?;
    let Some(metadata_json) = metadata_json else {
        return Ok(false);
    };
    let parsed = serde_json::from_str::<Value>(&metadata_json).unwrap_or_else(|_| Value::Null);
    Ok(parsed
        .get("preferredReplyModality")
        .and_then(Value::as_str)
        .is_some_and(|value| value.eq_ignore_ascii_case("voice")))
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
    normalized.is_empty() || normalized == "(no subject)" || normalized == "(ohne betreff)"
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

pub(crate) fn open_channel_db(path: &Path) -> Result<Connection> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("failed to create db parent {}", parent.display()))?;
    }
    #[cfg(test)]
    record_channel_db_open_for_tests(path);
    let conn = Connection::open(path)
        .with_context(|| format!("failed to open channel db {}", path.display()))?;
    conn.busy_timeout(crate::persistence::sqlite_busy_timeout_duration())
        .context("failed to configure SQLite busy_timeout for channels")?;
    ensure_schema_once(path, &conn)?;
    ensure_open_routing_rows_once(path, &conn)?;
    Ok(conn)
}

fn ensure_schema_once(path: &Path, conn: &Connection) -> Result<()> {
    let key = channel_schema_cache_key(path);
    let ready = CHANNEL_SCHEMA_READY.get_or_init(|| Mutex::new(HashSet::new()));
    let mut ready = ready
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if ready.contains(&key) {
        return Ok(());
    }
    ensure_schema(conn)?;
    #[cfg(test)]
    record_channel_schema_ensure_for_tests(&key);
    ready.insert(key);
    Ok(())
}

fn ensure_open_routing_rows_once(path: &Path, conn: &Connection) -> Result<()> {
    let key = channel_schema_cache_key(path);
    let ready = CHANNEL_OPEN_ROUTING_READY.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut ready = ready
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let stamp = channel_routing_cache_stamp(path);
    if ready.get(&key) == Some(&stamp) {
        return Ok(());
    }
    ensure_routing_rows_for_inbound(conn)?;
    #[cfg(test)]
    record_channel_open_routing_ensure_for_tests(&key);
    ready.insert(key, channel_routing_cache_stamp(path));
    Ok(())
}

#[cfg(unix)]
fn channel_schema_cache_key(path: &Path) -> ChannelSchemaCacheKey {
    let canonical = fs::canonicalize(path).unwrap_or_else(|_| absolute_channel_db_path(path));
    let metadata = fs::metadata(&canonical)
        .or_else(|_| fs::metadata(path))
        .ok();
    let (device, inode) = metadata
        .map(|metadata| (metadata.dev(), metadata.ino()))
        .unwrap_or((0, 0));
    (canonical, device, inode)
}

#[cfg(not(unix))]
fn channel_schema_cache_key(path: &Path) -> ChannelSchemaCacheKey {
    fs::canonicalize(path).unwrap_or_else(|_| absolute_channel_db_path(path))
}

fn queue_task_list_cache_key(
    path: &Path,
    statuses: &[String],
    limit: usize,
) -> QueueTaskListCacheKey {
    QueueTaskListCacheKey {
        database: channel_schema_cache_key(path),
        statuses: statuses.to_vec(),
        limit,
    }
}

fn cached_queue_task_list(
    key: &QueueTaskListCacheKey,
    stamp: &QueueTaskListCacheStamp,
) -> Option<Vec<QueueTaskView>> {
    let cache = QUEUE_TASK_LIST_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache
        .get(key)
        .filter(|entry| &entry.stamp == stamp)
        .map(|entry| entry.tasks.clone())
}

fn store_queue_task_list_cache(
    key: QueueTaskListCacheKey,
    stamp: QueueTaskListCacheStamp,
    tasks: Vec<QueueTaskView>,
) {
    let cache = QUEUE_TASK_LIST_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if cache.len() >= QUEUE_TASK_LIST_CACHE_MAX_ENTRIES && !cache.contains_key(&key) {
        cache.clear();
    }
    cache.insert(key, QueueTaskListCacheEntry { stamp, tasks });
}

fn queue_task_count_cache_key(path: &Path, statuses: &[String]) -> QueueTaskCountCacheKey {
    QueueTaskCountCacheKey {
        database: channel_schema_cache_key(path),
        statuses: statuses.to_vec(),
    }
}

fn cached_queue_task_count(
    key: &QueueTaskCountCacheKey,
    stamp: &QueueTaskListCacheStamp,
) -> Option<usize> {
    let cache = QUEUE_TASK_COUNT_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    cache
        .get(key)
        .filter(|entry| &entry.stamp == stamp)
        .map(|entry| entry.count)
}

fn store_queue_task_count_cache(
    key: QueueTaskCountCacheKey,
    stamp: QueueTaskListCacheStamp,
    count: usize,
) {
    let cache = QUEUE_TASK_COUNT_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut cache = cache
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if cache.len() >= QUEUE_TASK_COUNT_CACHE_MAX_ENTRIES && !cache.contains_key(&key) {
        cache.clear();
    }
    cache.insert(key, QueueTaskCountCacheEntry { stamp, count });
}

fn absolute_channel_db_path(path: &Path) -> PathBuf {
    if path.is_absolute() {
        return path.to_path_buf();
    }
    std::env::current_dir()
        .map(|cwd| cwd.join(path))
        .unwrap_or_else(|_| path.to_path_buf())
}

fn channel_routing_cache_stamp(path: &Path) -> ChannelRoutingCacheStamp {
    (
        channel_file_size_stamp(path),
        channel_file_size_stamp(&sqlite_sidecar_path(path, "-wal")),
        channel_file_size_stamp(&sqlite_sidecar_path(path, "-journal")),
    )
}

fn queue_task_list_cache_stamp(path: &Path) -> QueueTaskListCacheStamp {
    match queue_task_projection_clock_stamp(path) {
        Ok(stamp) => stamp,
        Err(_) => QueueTaskListCacheStamp::File {
            main: channel_file_change_stamp(path),
            wal: channel_file_change_stamp(&sqlite_sidecar_path(path, "-wal")),
            journal: channel_file_change_stamp(&sqlite_sidecar_path(path, "-journal")),
        },
    }
}

fn queue_task_projection_clock_stamp(path: &Path) -> Result<QueueTaskListCacheStamp> {
    let Some(conn) = open_channel_db_read_only(path)? else {
        return Ok(QueueTaskListCacheStamp::ProjectionClock {
            database_exists: false,
            clock_exists: false,
            version: 0,
            message_count: 0,
            routing_count: 0,
            updated_at: String::new(),
        });
    };
    let clock_exists = channel_projection_tables_exist(&conn, &["communication_projection_clock"])?;
    if !clock_exists {
        return Ok(QueueTaskListCacheStamp::ProjectionClock {
            database_exists: true,
            clock_exists: false,
            version: 0,
            message_count: 0,
            routing_count: 0,
            updated_at: String::new(),
        });
    }
    let (version, message_count, routing_count, updated_at) = conn.query_row(
        r#"
        SELECT version, message_count, routing_count, updated_at
        FROM communication_projection_clock
        WHERE id = 1
        "#,
        [],
        |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
            ))
        },
    )?;
    Ok(QueueTaskListCacheStamp::ProjectionClock {
        database_exists: true,
        clock_exists: true,
        version,
        message_count: non_negative_i64_to_usize(message_count),
        routing_count: non_negative_i64_to_usize(routing_count),
        updated_at,
    })
}

fn channel_file_size_stamp(path: &Path) -> u64 {
    fs::metadata(path)
        .map(|metadata| metadata.len())
        .unwrap_or(0)
}

fn channel_file_change_stamp(path: &Path) -> ChannelFileChangeStamp {
    let Ok(metadata) = fs::metadata(path) else {
        return (0, 0);
    };
    let modified_at = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    (metadata.len(), modified_at)
}

fn sqlite_sidecar_path(path: &Path, suffix: &str) -> PathBuf {
    let mut value = path.as_os_str().to_os_string();
    value.push(suffix);
    PathBuf::from(value)
}

#[cfg(test)]
fn record_channel_schema_ensure_for_tests(key: &ChannelSchemaCacheKey) {
    let counts = CHANNEL_SCHEMA_ENSURE_COUNTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *counts.entry(key.clone()).or_insert(0) += 1;
}

#[cfg(test)]
fn channel_schema_ensure_count_for_tests(path: &Path) -> usize {
    let key = channel_schema_cache_key(path);
    let Some(counts) = CHANNEL_SCHEMA_ENSURE_COUNTS.get() else {
        return 0;
    };
    let counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    counts.get(&key).copied().unwrap_or(0)
}

#[cfg(test)]
fn record_channel_open_routing_ensure_for_tests(key: &ChannelSchemaCacheKey) {
    let counts = CHANNEL_OPEN_ROUTING_ENSURE_COUNTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *counts.entry(key.clone()).or_insert(0) += 1;
}

#[cfg(test)]
fn channel_open_routing_ensure_count_for_tests(path: &Path) -> usize {
    let key = channel_schema_cache_key(path);
    let Some(counts) = CHANNEL_OPEN_ROUTING_ENSURE_COUNTS.get() else {
        return 0;
    };
    let counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    counts.get(&key).copied().unwrap_or(0)
}

#[cfg(test)]
fn record_channel_db_open_for_tests(path: &Path) {
    let counts = CHANNEL_DB_OPEN_CALL_COUNTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *counts.entry(path.to_path_buf()).or_insert(0) += 1;
}

#[cfg(test)]
pub(crate) fn reset_channel_db_open_count_for_tests(path: &Path) {
    if let Some(counts) = CHANNEL_DB_OPEN_CALL_COUNTS.get() {
        counts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .remove(path);
    }
}

#[cfg(test)]
pub(crate) fn channel_db_open_count_for_tests(path: &Path) -> usize {
    let Some(counts) = CHANNEL_DB_OPEN_CALL_COUNTS.get() else {
        return 0;
    };
    let counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    counts.get(path).copied().unwrap_or(0)
}

#[cfg(test)]
fn record_queue_task_list_cache_miss_for_tests(key: &QueueTaskListCacheKey) {
    let counts = QUEUE_TASK_LIST_CACHE_MISS_COUNTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *counts.entry(key.clone()).or_insert(0) += 1;
}

#[cfg(test)]
fn queue_task_list_cache_miss_count_for_tests(
    path: &Path,
    statuses: &[String],
    limit: usize,
) -> usize {
    let key = queue_task_list_cache_key(path, statuses, limit);
    let Some(counts) = QUEUE_TASK_LIST_CACHE_MISS_COUNTS.get() else {
        return 0;
    };
    let counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    counts.get(&key).copied().unwrap_or(0)
}

#[cfg(test)]
fn record_queue_task_count_cache_miss_for_tests(key: &QueueTaskCountCacheKey) {
    let counts = QUEUE_TASK_COUNT_CACHE_MISS_COUNTS.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *counts.entry(key.clone()).or_insert(0) += 1;
}

#[cfg(test)]
fn queue_task_count_cache_miss_count_for_tests(path: &Path, statuses: &[String]) -> usize {
    let key = queue_task_count_cache_key(path, statuses);
    let Some(counts) = QUEUE_TASK_COUNT_CACHE_MISS_COUNTS.get() else {
        return 0;
    };
    let counts = counts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    counts.get(&key).copied().unwrap_or(0)
}

fn open_channel_db_read_only(path: &Path) -> Result<Option<Connection>> {
    if !path.exists() {
        return Ok(None);
    }
    let conn = Connection::open_with_flags(path, OpenFlags::SQLITE_OPEN_READ_ONLY)
        .with_context(|| format!("failed to open channel db read-only {}", path.display()))?;
    conn.busy_timeout(crate::persistence::sqlite_busy_timeout_duration())
        .context("failed to configure SQLite busy_timeout for read-only channels")?;
    conn.execute_batch("PRAGMA query_only = ON;")
        .context("failed to configure read-only channel projection")?;
    Ok(Some(conn))
}

fn channel_projection_tables_exist(conn: &Connection, tables: &[&str]) -> Result<bool> {
    for table in tables {
        let exists = conn
            .query_row(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?1 LIMIT 1",
                params![table],
                |row| row.get::<_, i64>(0),
            )
            .optional()
            .with_context(|| format!("failed to inspect channel projection table {table}"))?
            .is_some();
        if !exists {
            return Ok(false);
        }
    }
    Ok(true)
}

fn empty_business_os_projection(collection: &str) -> Value {
    json!({
        "ok": true,
        "collection": collection,
        "documents": [],
        "count": 0,
        "since_ms": 0,
    })
}

fn ensure_schema(conn: &Connection) -> Result<()> {
    let busy_timeout_ms = crate::persistence::sqlite_busy_timeout_millis();
    conn.execute_batch(&format!(
        r#"
        PRAGMA journal_mode=WAL;
        PRAGMA busy_timeout={busy_timeout_ms};

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

        CREATE INDEX IF NOT EXISTS idx_communication_messages_email_folder_remote
            ON communication_messages(channel, account_key, folder_hint, remote_id);

        CREATE INDEX IF NOT EXISTS idx_communication_messages_queue_business_command_valid
            ON communication_messages(json_extract(metadata_json, '$.business_os_command_id'), observed_at DESC)
            WHERE channel = 'queue' AND direction = 'inbound' AND json_valid(metadata_json);

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
            first_pending_at TEXT,
            lease_expires_at TEXT,
            lease_worker_id TEXT,
            failure_class TEXT,
            failure_attempt_count INTEGER NOT NULL DEFAULT 0,
            retry_not_before TEXT,
            priority_time_credit_hours INTEGER NOT NULL DEFAULT 0,
            hold_reason TEXT,
            wait_entity_type TEXT,
            wait_entity_id TEXT,
            acked_at TEXT,
            last_error TEXT,
            updated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_communication_routing_status_owner
            ON communication_routing_state(route_status, lease_owner, leased_at, updated_at);

        CREATE TABLE IF NOT EXISTS communication_projection_clock (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            version INTEGER NOT NULL,
            account_count INTEGER NOT NULL,
            thread_count INTEGER NOT NULL,
            message_count INTEGER NOT NULL,
            routing_count INTEGER NOT NULL,
            updated_at TEXT NOT NULL
        );

        INSERT INTO communication_projection_clock (
            id, version, account_count, thread_count, message_count, routing_count, updated_at
        )
        SELECT
            1,
            0,
            (SELECT COUNT(*) FROM communication_accounts),
            (SELECT COUNT(*) FROM communication_threads),
            (SELECT COUNT(*) FROM communication_messages),
            (SELECT COUNT(*) FROM communication_routing_state),
            strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
        WHERE NOT EXISTS (
            SELECT 1 FROM communication_projection_clock WHERE id = 1
        );

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_accounts_insert
        AFTER INSERT ON communication_accounts
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                account_count = account_count + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_accounts_update
        AFTER UPDATE ON communication_accounts
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_accounts_delete
        AFTER DELETE ON communication_accounts
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                account_count = CASE
                    WHEN account_count > 0 THEN account_count - 1
                    ELSE 0
                END,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_threads_insert
        AFTER INSERT ON communication_threads
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                thread_count = thread_count + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_threads_update
        AFTER UPDATE ON communication_threads
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_threads_delete
        AFTER DELETE ON communication_threads
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                thread_count = CASE
                    WHEN thread_count > 0 THEN thread_count - 1
                    ELSE 0
                END,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_messages_insert
        AFTER INSERT ON communication_messages
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                message_count = message_count + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_messages_update
        AFTER UPDATE ON communication_messages
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_messages_delete
        AFTER DELETE ON communication_messages
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                message_count = CASE
                    WHEN message_count > 0 THEN message_count - 1
                    ELSE 0
                END,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_routing_insert
        AFTER INSERT ON communication_routing_state
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                routing_count = routing_count + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_routing_update
        AFTER UPDATE ON communication_routing_state
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TRIGGER IF NOT EXISTS trg_communication_projection_clock_routing_delete
        AFTER DELETE ON communication_routing_state
        BEGIN
            UPDATE communication_projection_clock
            SET version = version + 1,
                routing_count = CASE
                    WHEN routing_count > 0 THEN routing_count - 1
                    ELSE 0
                END,
                updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
            WHERE id = 1;
        END;

        CREATE TABLE IF NOT EXISTS business_command_aggregates (
            command_id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            payload_hash TEXT NOT NULL,
            module TEXT NOT NULL,
            command_type TEXT NOT NULL,
            record_id TEXT NOT NULL DEFAULT '',
            execution_mode TEXT NOT NULL CHECK(execution_mode IN ('control', 'queue')),
            execution_phase TEXT NOT NULL,
            terminal_status TEXT NOT NULL DEFAULT 'none',
            attempt INTEGER NOT NULL DEFAULT 0,
            projection_version INTEGER NOT NULL DEFAULT 1,
            intent_json TEXT NOT NULL,
            result_json TEXT,
            error_code TEXT,
            error_message TEXT,
            retryable INTEGER NOT NULL DEFAULT 0,
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_business_command_aggregates_state
            ON business_command_aggregates(execution_phase, updated_at_ms);

        CREATE TABLE IF NOT EXISTS business_command_task_links (
            command_id TEXT PRIMARY KEY,
            task_id TEXT NOT NULL UNIQUE,
            created_at_ms INTEGER NOT NULL,
            FOREIGN KEY(command_id) REFERENCES business_command_aggregates(command_id)
        );

        CREATE TABLE IF NOT EXISTS business_command_transitions (
            transition_id INTEGER PRIMARY KEY AUTOINCREMENT,
            command_id TEXT NOT NULL,
            projection_version INTEGER NOT NULL,
            from_phase TEXT NOT NULL,
            to_phase TEXT NOT NULL,
            terminal_status TEXT NOT NULL DEFAULT 'none',
            reason TEXT NOT NULL DEFAULT '',
            evidence_json TEXT NOT NULL DEFAULT '{{}}',
            created_at_ms INTEGER NOT NULL,
            UNIQUE(command_id, projection_version),
            FOREIGN KEY(command_id) REFERENCES business_command_aggregates(command_id)
        );

        CREATE TABLE IF NOT EXISTS business_command_effects (
            command_id TEXT NOT NULL,
            effect_key TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('claimed', 'completed', 'failed', 'uncertain')),
            result_json TEXT,
            error_message TEXT,
            claimed_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY(command_id, effect_key),
            FOREIGN KEY(command_id) REFERENCES business_command_aggregates(command_id)
        );

        CREATE TABLE IF NOT EXISTS business_command_sagas (
            saga_id TEXT PRIMARY KEY,
            command_id TEXT NOT NULL UNIQUE,
            saga_kind TEXT NOT NULL,
            phase TEXT NOT NULL CHECK(phase IN ('forward', 'compensating', 'completed', 'compensated', 'manual_intervention')),
            current_step INTEGER NOT NULL DEFAULT 0,
            total_steps INTEGER NOT NULL,
            compensation_status TEXT NOT NULL DEFAULT 'not_started',
            created_at_ms INTEGER NOT NULL,
            updated_at_ms INTEGER NOT NULL,
            FOREIGN KEY(command_id) REFERENCES business_command_aggregates(command_id)
        );

        CREATE TABLE IF NOT EXISTS business_command_saga_steps (
            saga_id TEXT NOT NULL,
            step_index INTEGER NOT NULL,
            step_name TEXT NOT NULL,
            forward_effect_key TEXT NOT NULL,
            compensation_effect_key TEXT NOT NULL,
            forward_status TEXT NOT NULL DEFAULT 'pending' CHECK(forward_status IN ('pending', 'claimed', 'completed', 'failed')),
            compensation_status TEXT NOT NULL DEFAULT 'not_required' CHECK(compensation_status IN ('not_required', 'pending', 'claimed', 'completed', 'failed')),
            forward_attempts INTEGER NOT NULL DEFAULT 0,
            compensation_attempts INTEGER NOT NULL DEFAULT 0,
            evidence_json TEXT NOT NULL DEFAULT '{{}}',
            error_message TEXT,
            updated_at_ms INTEGER NOT NULL,
            PRIMARY KEY(saga_id, step_index),
            UNIQUE(saga_id, forward_effect_key),
            UNIQUE(saga_id, compensation_effect_key),
            FOREIGN KEY(saga_id) REFERENCES business_command_sagas(saga_id)
        );

        CREATE TABLE IF NOT EXISTS business_app_action_snapshots (
            command_id TEXT PRIMARY KEY,
            module_id TEXT NOT NULL,
            action_name TEXT NOT NULL,
            definition_hash TEXT NOT NULL,
            definition_json TEXT NOT NULL,
            created_at_ms INTEGER NOT NULL,
            FOREIGN KEY(command_id) REFERENCES business_command_aggregates(command_id)
        );
        CREATE INDEX IF NOT EXISTS idx_business_app_action_snapshots_definition
            ON business_app_action_snapshots(module_id, action_name, definition_hash);

        CREATE TABLE IF NOT EXISTS business_command_results (
            command_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('succeeded', 'failed', 'cancelled')),
            user_reply TEXT NOT NULL DEFAULT '',
            artifacts_json TEXT NOT NULL DEFAULT '[]',
            writebacks_json TEXT NOT NULL DEFAULT '[]',
            claims_json TEXT NOT NULL DEFAULT '[]',
            error_json TEXT,
            review_status TEXT NOT NULL DEFAULT 'pending',
            validation_status TEXT NOT NULL DEFAULT 'pending',
            review_evidence_json TEXT NOT NULL DEFAULT '{{}}',
            created_at_ms INTEGER NOT NULL,
            reviewed_at_ms INTEGER,
            PRIMARY KEY(command_id, attempt),
            FOREIGN KEY(command_id) REFERENCES business_command_aggregates(command_id)
        );

        CREATE TABLE IF NOT EXISTS business_command_outbox (
            event_id TEXT PRIMARY KEY,
            command_id TEXT NOT NULL,
            projection_version INTEGER NOT NULL,
            destination TEXT NOT NULL CHECK(destination IN ('business-os', 'rxdb')),
            event_type TEXT NOT NULL,
            payload_json TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'pending' CHECK(status IN ('pending', 'delivered', 'failed', 'dead_letter')),
            attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at_ms INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            created_at_ms INTEGER NOT NULL,
            delivered_at_ms INTEGER,
            UNIQUE(command_id, projection_version, destination)
        );
        CREATE INDEX IF NOT EXISTS idx_business_command_outbox_delivery
            ON business_command_outbox(status, next_attempt_at_ms, created_at_ms);

        CREATE TABLE IF NOT EXISTS business_command_intake_failures (
            failure_id INTEGER PRIMARY KEY AUTOINCREMENT,
            command_id TEXT NOT NULL,
            attempt INTEGER NOT NULL,
            error_message TEXT NOT NULL,
            exhausted INTEGER NOT NULL DEFAULT 0,
            observed_at_ms INTEGER NOT NULL,
            resolved_at_ms INTEGER,
            UNIQUE(command_id, attempt)
        );
        CREATE INDEX IF NOT EXISTS idx_business_command_intake_failures_open
            ON business_command_intake_failures(command_id, resolved_at_ms, attempt);

        CREATE TABLE IF NOT EXISTS communication_founder_reply_reviews (
            approval_key TEXT PRIMARY KEY,
            inbound_message_key TEXT NOT NULL,
            action_digest TEXT NOT NULL,
            action_json TEXT NOT NULL,
            body_sha256 TEXT NOT NULL,
            reviewer TEXT NOT NULL,
            review_summary TEXT NOT NULL,
            approved_at TEXT NOT NULL,
            sent_at TEXT,
            send_result_json TEXT NOT NULL DEFAULT '{{}}',
            terminal_no_send INTEGER NOT NULL DEFAULT 0,
            UNIQUE(inbound_message_key, action_digest)
        );

        CREATE INDEX IF NOT EXISTS idx_founder_reply_reviews_inbound
            ON communication_founder_reply_reviews(inbound_message_key, sent_at);

        CREATE TABLE IF NOT EXISTS owner_profiles (
            owner_key TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#,
    ))
    .context("failed to ensure channel schema")?;
    ensure_terminal_no_send_column(conn)?;
    ensure_routing_state_hardening_columns(conn)?;
    Ok(())
}

fn ensure_routing_state_hardening_columns(conn: &Connection) -> Result<()> {
    for (column, definition) in [
        ("first_pending_at", "TEXT"),
        ("lease_expires_at", "TEXT"),
        ("lease_worker_id", "TEXT"),
        ("failure_class", "TEXT"),
        ("failure_attempt_count", "INTEGER NOT NULL DEFAULT 0"),
        ("retry_not_before", "TEXT"),
        ("priority_time_credit_hours", "INTEGER NOT NULL DEFAULT 0"),
        ("hold_reason", "TEXT"),
        ("wait_entity_type", "TEXT"),
        ("wait_entity_id", "TEXT"),
    ] {
        let exists: i64 = conn.query_row(
            "SELECT EXISTS(SELECT 1 FROM pragma_table_info('communication_routing_state') WHERE name=?1)",
            params![column],
            |row| row.get(0),
        )?;
        if exists == 0 {
            if let Err(err) = conn.execute_batch(&format!(
                "ALTER TABLE communication_routing_state ADD COLUMN {column} {definition};"
            )) {
                if !is_duplicate_column_error(&err) {
                    return Err(err).with_context(|| {
                        format!("failed to add communication_routing_state.{column}")
                    });
                }
            }
        }
    }
    conn.execute(
        r#"
        UPDATE communication_routing_state
        SET first_pending_at=COALESCE(
                first_pending_at,
                (SELECT COALESCE(m.external_created_at, m.observed_at)
                 FROM communication_messages m
                 WHERE m.message_key=communication_routing_state.message_key),
                updated_at
            ),
            lease_expires_at=CASE
                WHEN route_status='leased' AND lease_expires_at IS NULL
                THEN datetime(leased_at, '+15 minutes')
                ELSE lease_expires_at
            END
        WHERE first_pending_at IS NULL
           OR (route_status='leased' AND lease_expires_at IS NULL)
        "#,
        [],
    )?;
    Ok(())
}

/// Add the `terminal_no_send` column to
/// `communication_founder_reply_reviews` on existing databases that
/// were created before the NO-SEND verdict was a structured field.
/// New databases pick the column up from the CREATE TABLE statement
/// in this same migration block. This is idempotent: we probe via
/// `pragma_table_info` and only ALTER when the column is missing.
/// True when a rusqlite error is SQLite's "duplicate column name" error, which
/// is raised when an `ALTER TABLE ... ADD COLUMN` targets a column that already
/// exists. A probe-then-ALTER migration can hit this when a concurrent process
/// added the column between the probe and the ALTER, in which case the desired
/// end state already holds and the error is benign.
fn is_duplicate_column_error(err: &rusqlite::Error) -> bool {
    err.to_string()
        .to_ascii_lowercase()
        .contains("duplicate column name")
}

fn ensure_terminal_no_send_column(conn: &Connection) -> Result<()> {
    let column_exists: bool = conn
        .query_row(
            r#"
            SELECT EXISTS(
                SELECT 1
                FROM pragma_table_info('communication_founder_reply_reviews')
                WHERE name = 'terminal_no_send'
            )
            "#,
            [],
            |row| row.get::<_, i64>(0),
        )
        .map(|value| value != 0)
        .unwrap_or(false);
    if !column_exists {
        if let Err(err) = conn.execute(
            "ALTER TABLE communication_founder_reply_reviews ADD COLUMN terminal_no_send INTEGER NOT NULL DEFAULT 0",
            [],
        ) {
            // Cross-process race: another writer may have added the column
            // between the probe above and this ALTER. A duplicate-column error
            // means the column now exists, which is exactly the desired end
            // state, so tolerate it instead of failing open_channel_db.
            if !is_duplicate_column_error(&err) {
                return Err(err).context(
                    "failed to add terminal_no_send column to communication_founder_reply_reviews",
                );
            }
        }
    }
    Ok(())
}

pub(crate) fn ensure_routing_rows_for_inbound(conn: &Connection) -> Result<()> {
    // Historical auto-handle rule: inbound messages whose external timestamp
    // predates the communication account's creation are marked as already
    // handled so we don't re-process mailbox history at first boot. The
    // synthetic `queue` and `tui` channels are programmatic — work items are
    // created after the account exists and must stay `pending` until leased —
    // so they are excluded from the pre-account auto-handle.
    let mut statement = conn.prepare(
        r#"
        SELECT
            m.message_key,
            CASE
                WHEN m.direction = 'outbound' THEN 'handled'
                WHEN m.trust_level = 'system_probe' THEN 'handled'
                WHEN m.channel IN ('queue', 'tui') THEN 'pending'
                WHEN m.channel = 'teams'
                     AND m.direction = 'inbound'
                     AND a.created_at IS NOT NULL
                     AND m.external_created_at <= a.created_at
                     AND datetime(m.external_created_at) < datetime('now', '-24 hours') THEN 'handled'
                WHEN m.direction = 'inbound'
                     AND m.channel <> 'teams'
                     AND a.created_at IS NOT NULL
                     AND m.external_created_at <= a.created_at THEN 'handled'
	                ELSE 'pending'
            END,
            CASE
                WHEN m.direction = 'outbound' OR m.trust_level = 'system_probe' THEN m.observed_at
                WHEN m.channel IN ('queue', 'tui') THEN NULL
                WHEN m.channel = 'teams'
                     AND m.direction = 'inbound'
                     AND a.created_at IS NOT NULL
                     AND m.external_created_at <= a.created_at
                     AND datetime(m.external_created_at) < datetime('now', '-24 hours') THEN m.observed_at
                WHEN m.direction = 'inbound'
                     AND m.channel <> 'teams'
                     AND a.created_at IS NOT NULL
                     AND m.external_created_at <= a.created_at THEN m.observed_at
	                ELSE NULL
            END,
            m.observed_at
        FROM communication_messages m
        LEFT JOIN communication_accounts a ON a.account_key = m.account_key
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE r.message_key IS NULL
        "#,
    )?;
    let rows = statement.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<String>>(2)?,
            row.get::<_, String>(3)?,
        ))
    })?;
    let missing = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    for (message_key, route_status, acked_at, updated_at) in missing {
        let previous_route_status = current_queue_route_status(conn, &message_key)?;
        enforce_queue_route_status_transition(
            conn,
            &message_key,
            &previous_route_status,
            &route_status,
            "ctox-routing-backfill",
            "ensure_routing_rows_for_inbound",
        )?;
        conn.execute(
            r#"
            INSERT INTO communication_routing_state (
                message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at
            )
            VALUES (?1, ?2, NULL, NULL, ?3, NULL, ?4)
            ON CONFLICT(message_key) DO NOTHING
            "#,
            params![message_key, route_status, acked_at, updated_at],
        )?;
    }
    let mut statement = conn.prepare(
        r#"
        SELECT r.message_key, r.route_status
        FROM communication_routing_state r
        JOIN communication_messages m ON m.message_key = r.message_key
        WHERE m.direction = 'inbound'
          AND m.trust_level = 'system_probe'
          AND r.route_status <> 'handled'
        "#,
    )?;
    let rows = statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
    })?;
    let probe_updates = rows.collect::<rusqlite::Result<Vec<_>>>()?;
    drop(statement);
    for (message_key, previous_route_status) in probe_updates {
        enforce_queue_route_status_transition(
            conn,
            &message_key,
            &previous_route_status,
            "handled",
            "ctox-routing-backfill",
            "normalize_system_probe_messages",
        )?;
    }
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

fn ingest_tui_message(
    root: &Path,
    conn: &mut Connection,
    mut request: TuiIngestRequest,
) -> Result<Value> {
    let sanitized = secrets::auto_intake_prompt_secrets(root, &request.body)
        .context("failed to sanitize TUI secret-bearing input")?;
    if sanitized.auto_ingested_secrets > 0 {
        request.body = sanitized.sanitized_prompt;
        request.metadata = json!({
            "source": "ctox-channel-ingest-tui",
            "secret_sanitized": true,
            "auto_ingested_secrets": sanitized.auto_ingested_secrets,
            "suggested_skill": "secret-hygiene",
        });
    }
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
        WITH eligible AS (
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
                r.last_error,
                r.updated_at,
                MIN(COALESCE(r.first_pending_at, m.external_created_at)) OVER (
                    PARTITION BY m.thread_key
                ) AS thread_pending_since,
                MIN(r.priority_time_credit_hours) OVER (
                    PARTITION BY m.thread_key
                ) AS thread_priority_credit_hours,
                ROW_NUMBER() OVER (
                    PARTITION BY m.thread_key
                    ORDER BY
                        CASE WHEN m.channel = 'queue' THEN m.external_created_at END ASC,
                        CASE WHEN m.channel <> 'queue' THEN m.external_created_at END DESC,
                        CASE WHEN m.channel = 'queue' THEN m.observed_at END ASC,
                        CASE WHEN m.channel <> 'queue' THEN m.observed_at END DESC,
                        m.message_key DESC
                ) AS thread_rank
            FROM communication_messages m
            JOIN communication_routing_state r ON r.message_key = m.message_key
            WHERE m.direction = 'inbound'
              AND m.channel = ?1
              AND r.route_status = 'pending'
              AND (r.retry_not_before IS NULL OR datetime(r.retry_not_before) <= datetime('now'))
              AND (
                    json_extract(m.metadata_json, '$.not_before') IS NULL
                 OR json_extract(m.metadata_json, '$.not_before') = ''
                 OR json_extract(m.metadata_json, '$.not_before') <= strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
              )
        )
        SELECT
            message_key,
            channel,
            account_key,
            thread_key,
            remote_id,
            direction,
            folder_hint,
            sender_display,
            sender_address,
            subject,
            preview,
            body_text,
            status,
            seen,
            external_created_at,
            observed_at,
            metadata_json,
            route_status,
            lease_owner,
            leased_at,
            acked_at,
            last_error,
            updated_at
        FROM eligible
        WHERE thread_rank = 1
        ORDER BY
            CASE
                WHEN channel = 'tui' THEN datetime(thread_pending_since, '-24 hours')
                WHEN channel = 'queue' THEN datetime(thread_pending_since, '+1 hour')
                ELSE datetime(thread_pending_since, printf('%+d hours', thread_priority_credit_hours))
            END ASC,
            CASE WHEN channel = 'queue' THEN external_created_at END ASC,
            CASE WHEN channel <> 'queue' THEN external_created_at END DESC,
            CASE WHEN channel = 'queue' THEN observed_at END ASC,
            CASE WHEN channel <> 'queue' THEN observed_at END DESC,
            message_key DESC
        LIMIT ?3
        "#
    } else {
        r#"
        WITH eligible AS (
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
                r.last_error,
                r.updated_at,
                MIN(COALESCE(r.first_pending_at, m.external_created_at)) OVER (
                    PARTITION BY m.thread_key
                ) AS thread_pending_since,
                MIN(r.priority_time_credit_hours) OVER (
                    PARTITION BY m.thread_key
                ) AS thread_priority_credit_hours,
                ROW_NUMBER() OVER (
                    PARTITION BY m.thread_key
                    ORDER BY
                        CASE WHEN m.channel = 'queue' THEN m.external_created_at END ASC,
                        CASE WHEN m.channel <> 'queue' THEN m.external_created_at END DESC,
                        CASE WHEN m.channel = 'queue' THEN m.observed_at END ASC,
                        CASE WHEN m.channel <> 'queue' THEN m.observed_at END DESC,
                        m.message_key DESC
                ) AS thread_rank
            FROM communication_messages m
            JOIN communication_routing_state r ON r.message_key = m.message_key
            WHERE m.direction = 'inbound'
              AND r.route_status = 'pending'
              AND (r.retry_not_before IS NULL OR datetime(r.retry_not_before) <= datetime('now'))
              AND (
                    json_extract(m.metadata_json, '$.not_before') IS NULL
                 OR json_extract(m.metadata_json, '$.not_before') = ''
                 OR json_extract(m.metadata_json, '$.not_before') <= strftime('%Y-%m-%dT%H:%M:%SZ', 'now')
              )
        )
        SELECT
            message_key,
            channel,
            account_key,
            thread_key,
            remote_id,
            direction,
            folder_hint,
            sender_display,
            sender_address,
            subject,
            preview,
            body_text,
            status,
            seen,
            external_created_at,
            observed_at,
            metadata_json,
            route_status,
            lease_owner,
            leased_at,
            acked_at,
            last_error,
            updated_at
        FROM eligible
        WHERE thread_rank = 1
        ORDER BY
            CASE
                WHEN channel = 'tui' THEN datetime(thread_pending_since, '-24 hours')
                WHEN channel = 'queue' THEN datetime(thread_pending_since, '+1 hour')
                ELSE datetime(thread_pending_since, printf('%+d hours', thread_priority_credit_hours))
            END ASC,
            CASE WHEN channel = 'queue' THEN external_created_at END ASC,
            CASE WHEN channel <> 'queue' THEN external_created_at END DESC,
            CASE WHEN channel = 'queue' THEN observed_at END ASC,
            CASE WHEN channel <> 'queue' THEN observed_at END DESC,
            message_key DESC
        LIMIT ?2
        "#
    };

    // Hold a write lock for the whole check-then-act window so a concurrent
    // leaser on a different connection cannot steal a lease between our
    // eligibility SELECT and our UPDATE (lost-update). The lease UPDATE is a
    // check-and-set: its WHERE mirrors the eligibility predicate above, so a
    // losing racer flips 0 rows and we record neither the row nor a
    // core-transition proof for it.
    let tx =
        rusqlite::Transaction::new_unchecked(&*conn, rusqlite::TransactionBehavior::Immediate)?;
    let rows = {
        let mut statement = tx.prepare(sql)?;
        let mapped = if let Some(channel) = channel {
            statement.query_map(
                params![channel, lease_owner, limit as i64],
                map_channel_message_row,
            )?
        } else {
            statement.query_map(params![lease_owner, limit as i64], map_channel_message_row)?
        };
        mapped.collect::<rusqlite::Result<Vec<_>>>()?
    };
    let leased_at = now_iso_string();
    let lease_expires_at = (chrono::Utc::now() + chrono::Duration::minutes(15)).to_rfc3339();
    let mut taken = Vec::new();
    for mut item in rows {
        let updated = tx.execute(
            r#"INSERT INTO communication_routing_state (message_key, route_status, lease_owner, leased_at, first_pending_at, lease_expires_at, lease_worker_id, acked_at, last_error, updated_at)
               VALUES (?1, 'leased', ?2, ?3, ?3, ?4, NULL, NULL, NULL, ?3)
               ON CONFLICT(message_key) DO UPDATE SET route_status='leased', lease_owner=excluded.lease_owner, leased_at=excluded.leased_at, first_pending_at=COALESCE(communication_routing_state.first_pending_at, excluded.first_pending_at), lease_expires_at=excluded.lease_expires_at, lease_worker_id=NULL, retry_not_before=NULL, hold_reason=NULL, acked_at=NULL, updated_at=excluded.updated_at
               WHERE communication_routing_state.route_status = 'pending'"#,
            params![item.message_key, lease_owner, leased_at, lease_expires_at],
        )?;
        if updated != 0 {
            enforce_queue_route_status_transition(
                &tx,
                &item.message_key,
                &item.routing.route_status,
                "leased",
                lease_owner,
                "lease_messages",
            )?;
        }
        if updated == 0 {
            // Lost the race: another owner leased this key between our SELECT
            // and UPDATE. Skip the core-transition proof and the push so we
            // never return a message we did not actually lease.
            continue;
        }
        item.routing.route_status = "leased".to_string();
        item.routing.lease_owner = Some(lease_owner.to_string());
        item.routing.leased_at = Some(leased_at.clone());
        item.routing.updated_at = leased_at.clone();
        taken.push(item);
    }
    tx.commit()?;
    Ok(taken)
}

pub fn defer_messages_until(
    root: &Path,
    message_keys: &[String],
    not_before: &str,
    reason: &str,
) -> Result<usize> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    let mut updated = 0usize;
    for message_key in message_keys {
        let message_updated = conn.execute(
            r#"
            UPDATE communication_messages
            SET metadata_json = json_set(
                json_set(metadata_json, '$.not_before', ?2),
                '$.defer_reason',
                ?3
            )
            WHERE message_key = ?1
            "#,
            params![message_key, not_before, reason],
        )?;
        if message_updated != 0 {
            conn.execute(
                r#"
                UPDATE communication_routing_state
                SET retry_not_before = ?2,
                    hold_reason = ?3,
                    updated_at = ?4
                WHERE message_key = ?1
                "#,
                params![message_key, not_before, reason, now_iso_string()],
            )?;
        }
        updated += message_updated;
    }
    Ok(updated)
}

fn ack_messages(
    conn: &mut Connection,
    message_keys: &[String],
    status: &str,
    failure_note: Option<&str>,
    ack_reason: Option<&str>,
) -> Result<usize> {
    let tx = conn.unchecked_transaction()?;
    let updated = ack_messages_in_transaction(&tx, message_keys, status, failure_note, ack_reason)?;
    tx.commit()?;
    Ok(updated)
}

fn ack_messages_in_transaction(
    tx: &Transaction<'_>,
    message_keys: &[String],
    status: &str,
    failure_note: Option<&str>,
    ack_reason: Option<&str>,
) -> Result<usize> {
    let now = now_iso_string();
    let acked_at = if matches!(status, "handled" | "cancelled") {
        Some(now.as_str())
    } else {
        None
    };
    let failure_note = if status == "failed" {
        Some(
            failure_note
                .map(str::trim)
                .filter(|value| !value.is_empty())
                .context("failed queue ack requires a non-empty failure reason")?,
        )
    } else {
        None
    };
    let mut updated = 0usize;
    for message_key in message_keys {
        let previous_route_status = current_queue_route_status(&tx, message_key)?;
        enforce_queue_route_status_transition(
            &tx,
            message_key,
            &previous_route_status,
            status,
            "ctox-queue-ack",
            ack_reason.or(failure_note).unwrap_or("ack_messages"),
        )?;
        let routing_updates = tx.execute(
            r#"
            INSERT INTO communication_routing_state (
                message_key, route_status, lease_owner, leased_at, acked_at, last_error, updated_at
            )
            SELECT ?1, ?2, NULL, NULL, ?3, ?4, ?5
            FROM communication_messages
            WHERE message_key = ?1
            ON CONFLICT(message_key) DO UPDATE SET
                route_status=excluded.route_status,
                lease_owner=NULL,
                leased_at=NULL,
                lease_worker_id=NULL,
                acked_at=excluded.acked_at,
                last_error=excluded.last_error,
                updated_at=excluded.updated_at
            "#,
            params![message_key, status, acked_at, failure_note, now],
        )?;
        if routing_updates == 0 {
            continue;
        }
        updated += routing_updates;
        tx.execute(
            "UPDATE communication_messages SET seen = 1 WHERE message_key = ?1",
            params![message_key],
        )?;
    }
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
            r.last_error,
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
            r.last_error,
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
            r.last_error,
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
    let normalized_query = format!("%{}%", search_query_seed(query));
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
            r.last_error,
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

fn search_query_seed(query: &str) -> String {
    let compact = query.trim().to_ascii_lowercase();
    compact
        .split(|ch: char| !ch.is_alphanumeric() && ch != '_' && ch != '-')
        .find(|token| token.len() >= 3)
        .unwrap_or(compact.as_str())
        .to_string()
}

fn build_communication_context(
    conn: &Connection,
    thread_key: &str,
    query: Option<&str>,
    sender: Option<&str>,
    limit: usize,
) -> Result<CommunicationContextView> {
    let thread_messages = list_thread_messages(conn, thread_key, limit)?;
    let latest_subject = thread_messages
        .iter()
        .find(|item| !item.subject.trim().is_empty())
        .map(|item| item.subject.clone());
    let mut related_messages = Vec::new();
    if let Some(query) = query.map(str::trim).filter(|value| !value.is_empty()) {
        let mut seen = std::collections::BTreeSet::new();
        for message in search_messages(conn, query, None, None, limit)?
            .into_iter()
            .chain(
                sender
                    .map(|value| search_messages(conn, query, None, Some(value), limit))
                    .transpose()?
                    .unwrap_or_default()
                    .into_iter(),
            )
        {
            if seen.insert(message.message_key.clone()) {
                related_messages.push(message);
            }
        }
    }
    related_messages.retain(|item| item.thread_key != thread_key);
    let latest_inbound = thread_messages
        .iter()
        .find(|item| item.direction == "inbound")
        .map(|item| candidate_from_message("latest_inbound", item));
    let latest_outbound = thread_messages
        .iter()
        .find(|item| item.direction == "outbound")
        .map(|item| candidate_from_message("latest_outbound", item));
    let mut candidate_blockers = collect_candidates(&thread_messages, &related_messages, "blocker");
    let mut candidate_promises = collect_candidates(&thread_messages, &related_messages, "promise");
    let open_owner_questions = collect_open_owner_questions(&thread_messages);
    candidate_blockers.truncate(limit.min(8));
    candidate_promises.truncate(limit.min(8));
    Ok(CommunicationContextView {
        thread_key: thread_key.to_string(),
        latest_subject,
        latest_inbound,
        latest_outbound,
        thread_messages,
        related_messages,
        candidate_blockers,
        candidate_promises,
        open_owner_questions,
    })
}

fn collect_candidates(
    thread_messages: &[ChannelMessageView],
    related_messages: &[ChannelMessageView],
    candidate_kind: &str,
) -> Vec<CommunicationStateCandidate> {
    let mut out = Vec::new();
    for message in thread_messages.iter().chain(related_messages.iter()) {
        if message.direction != "outbound" {
            continue;
        }
        let body = format!(
            "{}\n{}\n{}",
            message.subject.to_ascii_lowercase(),
            message.preview.to_ascii_lowercase(),
            message.body_text.to_ascii_lowercase()
        );
        let is_match = match candidate_kind {
            "blocker" => {
                body.contains("blocked")
                    || body.contains("blocker")
                    || body.contains("need ")
                    || body.contains("missing ")
                    || body.contains("requires ")
                    || body.contains("cannot ")
            }
            "promise" => {
                body.contains("next step")
                    || body.contains("i will")
                    || body.contains("i'll")
                    || body.contains("follow-up")
                    || body.contains("queued")
                    || body.contains("review")
                    || body.contains("continue")
            }
            _ => false,
        };
        if is_match {
            out.push(candidate_from_message(candidate_kind, message));
        }
    }
    out
}

fn collect_open_owner_questions(
    thread_messages: &[ChannelMessageView],
) -> Vec<CommunicationStateCandidate> {
    let latest_outbound_at = thread_messages
        .iter()
        .find(|item| item.direction == "outbound")
        .map(|item| item.external_created_at.clone());
    thread_messages
        .iter()
        .filter(|item| item.direction == "inbound")
        .filter(|item| {
            let text = format!("{}\n{}", item.subject, item.body_text);
            text.contains('?')
                || text.to_ascii_lowercase().contains("please")
                || text.to_ascii_lowercase().contains("can you")
        })
        .filter(|item| {
            latest_outbound_at
                .as_ref()
                .map(|outbound| item.external_created_at >= *outbound)
                .unwrap_or(true)
        })
        .map(|item| candidate_from_message("open_question", item))
        .collect()
}

fn candidate_from_message(kind: &str, message: &ChannelMessageView) -> CommunicationStateCandidate {
    CommunicationStateCandidate {
        kind: kind.to_string(),
        message_key: message.message_key.clone(),
        channel: message.channel.clone(),
        thread_key: message.thread_key.clone(),
        created_at: message.external_created_at.clone(),
        summary: preview_text(&message.body_text, &message.subject),
    }
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
            last_error: row.get(21)?,
            updated_at: row.get(22)?,
        },
    })
}

fn routed_inbound_message_from_view(item: ChannelMessageView) -> RoutedInboundMessage {
    let preferred_reply_modality = item
        .metadata
        .get("preferredReplyModality")
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let workspace_root =
        workspace_root_from_queue_metadata_or_prompt(&item.metadata, &item.body_text);
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
        workspace_root,
        metadata: item.metadata,
        preferred_reply_modality,
    }
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
            r.last_error,
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

fn list_queue_tasks_from_conn_with_statuses(
    conn: &Connection,
    statuses: &[String],
    limit: usize,
) -> Result<Vec<QueueTaskView>> {
    let placeholders = (0..statuses.len())
        .map(|index| format!("?{}", index + 3))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
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
            r.last_error,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = ?1
          AND m.direction = 'inbound'
          AND lower(COALESCE(r.route_status, 'pending')) IN ({placeholders})
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
        "#
    );
    let mut values = Vec::with_capacity(statuses.len() + 2);
    values.push(SqlValue::Text(QUEUE_CHANNEL_NAME.to_string()));
    values.push(SqlValue::Integer(limit as i64));
    values.extend(statuses.iter().cloned().map(SqlValue::Text));
    let mut statement = conn.prepare(&sql)?;
    let rows = statement.query_map(params_from_iter(values), map_channel_message_row)?;
    rows.collect::<rusqlite::Result<Vec<_>>>()
        .map_err(anyhow::Error::from)?
        .into_iter()
        .map(queue_task_from_message)
        .collect()
}

fn count_queue_tasks_from_conn_with_statuses(
    conn: &Connection,
    statuses: &[String],
) -> Result<usize> {
    let placeholders = (0..statuses.len())
        .map(|index| format!("?{}", index + 2))
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        r#"
        SELECT COUNT(*)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = ?1
          AND m.direction = 'inbound'
          AND lower(COALESCE(r.route_status, 'pending')) IN ({placeholders})
        "#
    );
    let mut values = Vec::with_capacity(statuses.len() + 1);
    values.push(SqlValue::Text(QUEUE_CHANNEL_NAME.to_string()));
    values.extend(statuses.iter().cloned().map(SqlValue::Text));
    let count: i64 = conn.query_row(&sql, params_from_iter(values), |row| row.get(0))?;
    Ok(count.max(0) as usize)
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
            r.last_error,
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

pub(crate) fn load_queue_tasks_by_message_key_from_conn(
    conn: &Connection,
    message_keys: &[String],
) -> Result<BTreeMap<String, QueueTaskView>> {
    let mut tasks = BTreeMap::new();
    if message_keys.is_empty() {
        return Ok(tasks);
    }
    for chunk in message_keys.chunks(500) {
        let placeholders = (0..chunk.len())
            .map(|index| format!("?{}", index + 2))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
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
                r.last_error,
                COALESCE(r.updated_at, m.observed_at)
            FROM communication_messages m
            LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
            WHERE m.channel = ?1
              AND m.direction = 'inbound'
              AND m.message_key IN ({placeholders})
            "#
        );
        let mut values = Vec::with_capacity(chunk.len() + 1);
        values.push(SqlValue::Text(QUEUE_CHANNEL_NAME.to_string()));
        values.extend(chunk.iter().cloned().map(SqlValue::Text));
        let mut statement = conn.prepare(&sql)?;
        let rows = statement.query_map(params_from_iter(values), map_channel_message_row)?;
        for row in rows {
            let task = queue_task_from_message(row?)?;
            tasks.insert(task.message_key.clone(), task);
        }
    }
    Ok(tasks)
}

fn load_queue_task_for_business_os_command_from_conn(
    conn: &Connection,
    command_id: &str,
) -> Result<Option<QueueTaskView>> {
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
            r.last_error,
            COALESCE(r.updated_at, m.observed_at)
        FROM communication_messages m
        LEFT JOIN communication_routing_state r ON r.message_key = m.message_key
        WHERE m.channel = ?1
          AND m.direction = 'inbound'
          AND json_valid(m.metadata_json)
          AND json_extract(m.metadata_json, '$.business_os_command_id') = ?2
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
        LIMIT 1
        "#,
        params![QUEUE_CHANNEL_NAME, command_id],
        map_channel_message_row,
    )
    .optional()?
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
    let prompt = message.body_text;
    let workspace_root = workspace_root_from_queue_metadata_or_prompt(&message.metadata, &prompt);
    let route_status = message.routing.route_status.clone();
    let metadata_status_note = || {
        message
            .metadata
            .get("status_note")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
    };
    let status_note = if route_status == "failed" {
        message
            .routing
            .last_error
            .clone()
            .or_else(metadata_status_note)
    } else {
        metadata_status_note()
    };
    Ok(QueueTaskView {
        message_key: message.message_key,
        thread_key: message.thread_key,
        title: message.subject,
        prompt,
        workspace_root,
        ticket_self_work_id: message
            .metadata
            .get("ticket_self_work_id")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned),
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
        route_status,
        status_note,
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
    conn: &Connection,
    message_key: &str,
    route_status: &str,
    now: &str,
    actor: &str,
    reason: &str,
    status_note: Option<&str>,
) -> Result<()> {
    let route_status = canonical_queue_route_status(route_status)?;
    let failure_note = if route_status == "failed" {
        let note = status_note
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .context("failed queue route status requires a non-empty status_note/failure reason")?;
        Some(note)
    } else {
        None
    };
    let previous_route_status = current_queue_route_status(conn, message_key)?;
    let transition_reason = if route_status == "failed" {
        failure_note.unwrap_or(reason)
    } else {
        status_note
            .map(str::trim)
            .filter(|value| !value.is_empty())
            .unwrap_or(reason)
    };
    enforce_queue_route_status_transition(
        conn,
        message_key,
        &previous_route_status,
        &route_status,
        actor,
        transition_reason,
    )?;
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
        VALUES (?1, ?2, NULL, NULL, ?3, ?4, ?5)
        ON CONFLICT(message_key) DO UPDATE SET
            route_status=excluded.route_status,
            lease_owner=NULL,
            leased_at=NULL,
            lease_worker_id=NULL,
            lease_expires_at=CASE WHEN excluded.route_status='pending' THEN NULL ELSE communication_routing_state.lease_expires_at END,
            retry_not_before=CASE WHEN excluded.route_status='pending' THEN NULL ELSE communication_routing_state.retry_not_before END,
            hold_reason=CASE WHEN excluded.route_status='pending' THEN NULL ELSE communication_routing_state.hold_reason END,
            wait_entity_type=CASE WHEN excluded.route_status='pending' THEN NULL ELSE communication_routing_state.wait_entity_type END,
            wait_entity_id=CASE WHEN excluded.route_status='pending' THEN NULL ELSE communication_routing_state.wait_entity_id END,
            acked_at=excluded.acked_at,
            last_error=excluded.last_error,
            updated_at=excluded.updated_at
        "#,
        params![message_key, route_status, acked_at, failure_note, now],
    )?;
    Ok(())
}

pub(crate) fn current_queue_route_status(conn: &Connection, message_key: &str) -> Result<String> {
    conn.query_row(
        "SELECT route_status FROM communication_routing_state WHERE message_key = ?1 LIMIT 1",
        params![message_key],
        |row| row.get::<_, String>(0),
    )
    .optional()
    .map(|value| value.unwrap_or_else(|| "pending".to_string()))
    .map_err(anyhow::Error::from)
}

/// Legacy routing statuses that older builds wrote directly into
/// `communication_routing_state`. They stay readable as transition SOURCES
/// so normalization/healing can move such rows to canonical statuses, but
/// they are not writable targets. `Blocked` is the non-terminal state with
/// healing edges to Pending, Completed, Failed, and Superseded.
fn legacy_queue_route_status_core_state(route_status: &str) -> Option<CoreState> {
    match route_status.trim().to_ascii_lowercase().as_str() {
        "duplicate" | "blocked_sender" | "meeting_scheduled" => Some(CoreState::Blocked),
        _ => None,
    }
}

pub(crate) fn enforce_queue_route_status_transition(
    conn: &Connection,
    message_key: &str,
    from_route_status: &str,
    to_route_status: &str,
    actor: &str,
    reason: &str,
) -> Result<()> {
    let from_state = match queue_route_status_core_state(from_route_status) {
        Ok(state) => state,
        Err(err) => legacy_queue_route_status_core_state(from_route_status).ok_or(err)?,
    };
    let to_state = queue_route_status_core_state(to_route_status)?;
    if from_state == to_state {
        return Ok(());
    }
    if to_state == CoreState::Completed
        && queue_completed_has_terminal_success_proof(conn, message_key)?
    {
        return Ok(());
    }
    if to_state == CoreState::ReworkRequired && queue_rework_has_witness_proof(conn, message_key)? {
        return Ok(());
    }
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "from_route_status".to_string(),
        from_route_status.to_string(),
    );
    metadata.insert("to_route_status".to_string(), to_route_status.to_string());
    metadata.insert("reason".to_string(), reason.to_string());
    if to_state == CoreState::Failed {
        metadata.insert("failure_reason".to_string(), reason.to_string());
        metadata.insert(
            "failure_class".to_string(),
            "queue_route_failure".to_string(),
        );
    }
    if to_state == CoreState::Completed {
        if let Some(policy_proof) = queue_terminal_policy_proof(actor, reason) {
            metadata.insert("terminal_policy_proof".to_string(), policy_proof);
        }
    }
    enforce_core_transition(
        conn,
        &CoreTransitionRequest {
            entity_type: CoreEntityType::QueueItem,
            entity_id: message_key.to_string(),
            lane: RuntimeLane::P2MissionDelivery,
            from_state,
            to_state,
            event: queue_route_status_core_event(to_route_status),
            actor: actor.to_string(),
            evidence: CoreEvidenceRefs::default(),
            metadata,
        },
    )?;
    Ok(())
}

/// Root-based wrapper: does this queue item already carry an accepted
/// terminal-success Completed proof? Queue repair uses it to pre-classify a
/// `complete` action before it reaches the Completed gate, so an unproven
/// complete is refused-and-skipped instead of aborting the whole repair pass.
pub fn queue_complete_action_has_terminal_proof(root: &Path, message_key: &str) -> Result<bool> {
    let db_path = resolve_db_path(root, None);
    let conn = open_channel_db(&db_path)?;
    queue_completed_has_terminal_success_proof(&conn, message_key)
}

fn queue_completed_has_terminal_success_proof(
    conn: &Connection,
    message_key: &str,
) -> Result<bool> {
    ensure_core_transition_guard_schema(conn)?;
    let count = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM ctox_core_transition_proofs
        WHERE entity_type = 'QueueItem'
          AND entity_id = ?1
          AND to_state = 'Completed'
          AND accepted = 1
          AND (
                request_json LIKE '%"reviewed_work_terminal_success":"true"%'
             OR request_json LIKE '%"terminal_policy_proof"%'
          )
        "#,
        params![message_key],
        |row| row.get::<_, i64>(0),
    )?;
    Ok(count > 0)
}

fn queue_rework_has_witness_proof(conn: &Connection, message_key: &str) -> Result<bool> {
    ensure_core_transition_guard_schema(conn)?;
    let count = conn.query_row(
        r#"
        SELECT COUNT(*)
        FROM ctox_core_transition_proofs
        WHERE entity_type = 'QueueItem'
          AND entity_id = ?1
          AND to_state = 'ReworkRequired'
          AND accepted = 1
          AND (
                request_json LIKE '%"review_checkpoint":"true"%'
             OR request_json LIKE '%"validator_rework":"true"%'
          )
        "#,
        params![message_key],
        |row| row.get::<_, i64>(0),
    )?;
    Ok(count > 0)
}

fn queue_terminal_policy_proof(actor: &str, reason: &str) -> Option<String> {
    if actor == "business-command-terminal-owner" {
        // This actor is reachable only through
        // `transition_business_command_for_task`, which has already verified
        // the immutable typed result plus passed review and validation in the
        // same core transaction.
        return Some("policy:business-command-reviewed-terminal-success".to_string());
    }
    if actor == "ctox-queue-update" && reason.starts_with("business-os:terminal-success:") {
        return Some("policy:business-os-queued-command-terminal-success".to_string());
    }
    if actor == "ctox-queue-update" && reason.starts_with("appsec:terminal-success:") {
        return Some("policy:appsec-pipeline-stage-terminal-success".to_string());
    }
    match (actor, reason) {
        // Inbound mail fully handled by scheduling the requested meeting:
        // routing closes the message without a worker/review pass.
        ("ctox-queue-ack", "meeting_scheduled") => {
            Some("policy:meeting-scheduled-terminal-no-send".to_string())
        }
        // Inbound mail that only passively mentions a meeting; routing
        // closes it as non-work without a worker/review pass.
        ("ctox-queue-ack", "meeting_passive_mention") => {
            Some("policy:meeting-passive-inbound-terminal-no-send".to_string())
        }
        ("ctox-boot-reclassifier", "mark_historical_auto_submitted_inbound_handled") => {
            Some("policy:auto-submitted-inbound-terminal-no-send".to_string())
        }
        ("ctox-routing-backfill", "normalize_system_probe_messages") => {
            Some("policy:system-probe-inbound-terminal-no-send".to_string())
        }
        ("ctox-routing-backfill", "ensure_routing_rows_for_inbound") => {
            Some("policy:routing-backfill-non-work-terminal-no-send".to_string())
        }
        _ => None,
    }
}

fn queue_route_status_core_state(route_status: &str) -> Result<CoreState> {
    match route_status.trim().to_ascii_lowercase().as_str() {
        "" | "pending" => Ok(CoreState::Pending),
        "leased" => Ok(CoreState::Leased),
        "running" => Ok(CoreState::Running),
        "blocked" | "approval-nag-handled" => Ok(CoreState::Blocked),
        "review_rework" => Ok(CoreState::ReworkRequired),
        "failed" => Ok(CoreState::Failed),
        "handled" | "completed" => Ok(CoreState::Completed),
        "cancelled" | "superseded" => Ok(CoreState::Superseded),
        other => anyhow::bail!("queue route status is not mapped to core state machine: {other}"),
    }
}

fn queue_route_status_core_event(route_status: &str) -> CoreEvent {
    match route_status.trim().to_ascii_lowercase().as_str() {
        "leased" => CoreEvent::Lease,
        "pending" => CoreEvent::Release,
        "blocked" | "approval-nag-handled" => CoreEvent::Block,
        "review_rework" => CoreEvent::RequireRework,
        "failed" => CoreEvent::Fail,
        "cancelled" | "superseded" => CoreEvent::Supersede,
        "handled" | "completed" => CoreEvent::Complete,
        _ => CoreEvent::Retry,
    }
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
        "pending" | "leased" | "running" | "blocked" | "failed" | "handled" | "cancelled"
        | "review_rework" => {
            Ok(normalized)
        }
        _ => anyhow::bail!(
            "unsupported queue route status '{raw}' (expected pending|leased|running|blocked|failed|handled|cancelled|review_rework)"
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

fn normalize_workspace_root(raw: Option<&str>) -> Option<String> {
    raw.map(str::trim)
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
}

pub fn workspace_root_from_queue_metadata_or_prompt(
    metadata: &Value,
    prompt: &str,
) -> Option<String> {
    queue_metadata_object(metadata)
        .get("workspace_root")
        .and_then(Value::as_str)
        .and_then(|value| normalize_workspace_root(Some(value)))
        .or_else(|| legacy_workspace_root_from_prompt(prompt))
}

pub fn legacy_workspace_root_from_prompt(prompt: &str) -> Option<String> {
    for marker in [
        "Work only inside this workspace:",
        "Arbeite ausschließlich im Verzeichnis ",
        "Arbeite im Verzeichnis ",
        "Arbeite ausschließlich im Workspace ",
        "Arbeite im Workspace ",
    ] {
        if let Some(path) = extract_workspace_root_after_marker(prompt, marker) {
            return Some(path);
        }
    }
    None
}

fn extract_workspace_root_after_marker(prompt: &str, marker: &str) -> Option<String> {
    let start = prompt.find(marker)? + marker.len();
    let tail = prompt[start..].trim_start();
    let line = tail.lines().next()?.trim();
    let candidate = if let Some(stripped) = line.strip_prefix('/') {
        format!("/{stripped}")
    } else if let Some(index) = line.find('/') {
        line[index..].to_string()
    } else {
        return None;
    };
    let trimmed = candidate
        .trim_end_matches(|ch: char| matches!(ch, '.' | ',' | ';' | ':' | ')' | ']' | '"' | '\''));
    normalize_workspace_root(Some(trimmed))
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

pub(crate) struct UpsertMessage<'a> {
    pub message_key: &'a str,
    pub channel: &'a str,
    pub account_key: &'a str,
    pub thread_key: &'a str,
    pub remote_id: &'a str,
    pub direction: &'a str,
    pub folder_hint: &'a str,
    pub sender_display: &'a str,
    pub sender_address: &'a str,
    pub recipient_addresses_json: &'a str,
    pub cc_addresses_json: &'a str,
    pub bcc_addresses_json: &'a str,
    pub subject: &'a str,
    pub preview: &'a str,
    pub body_text: &'a str,
    pub body_html: &'a str,
    pub raw_payload_ref: &'a str,
    pub trust_level: &'a str,
    pub status: &'a str,
    pub seen: bool,
    pub has_attachments: bool,
    pub external_created_at: &'a str,
    pub observed_at: &'a str,
    pub metadata_json: &'a str,
}

pub(crate) struct CommunicationSyncRun<'a> {
    pub run_key: &'a str,
    pub channel: &'a str,
    pub account_key: &'a str,
    pub folder_hint: &'a str,
    pub started_at: &'a str,
    pub finished_at: &'a str,
    pub ok: bool,
    pub fetched_count: i64,
    pub stored_count: i64,
    pub error_text: &'a str,
    pub metadata_json: &'a str,
}

pub(crate) fn upsert_communication_message(
    conn: &mut Connection,
    message: UpsertMessage<'_>,
) -> Result<()> {
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

pub(crate) fn ensure_account(
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

fn sync_identity_profiles(
    conn: &mut Connection,
    settings: &BTreeMap<String, String>,
) -> Result<()> {
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

    let founder_roles = parse_founder_email_roles(settings);
    for founder_email in parse_founder_email_addresses(settings) {
        let founder_role = founder_roles
            .get(&founder_email)
            .cloned()
            .unwrap_or_else(|| "Founder".to_string());
        upsert_identity_profile(
            conn,
            &founder_email,
            &founder_email,
            json!({
                "email": founder_email,
                "role": "founder",
                "role_title": founder_role,
                "allow_admin_actions": true,
                "allow_sudo_actions": false,
                "mail_instruction_scope": "founder_strategic",
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

fn parse_founder_email_addresses(settings: &BTreeMap<String, String>) -> Vec<String> {
    let raw = settings
        .get("CTOX_FOUNDER_EMAIL_ADDRESSES")
        .map(String::as_str)
        .unwrap_or("");
    let mut seen = BTreeSet::new();
    raw.split(|ch| matches!(ch, '\n' | ',' | ';'))
        .map(str::trim)
        .filter(|value| !value.is_empty())
        .map(normalize_email_address)
        .filter(|value| !value.is_empty())
        .filter(|value| seen.insert(value.clone()))
        .collect()
}

fn parse_founder_email_roles(settings: &BTreeMap<String, String>) -> BTreeMap<String, String> {
    let raw = settings
        .get("CTOX_FOUNDER_EMAIL_ROLES")
        .map(String::as_str)
        .unwrap_or("");
    let mut roles = BTreeMap::new();
    for entry in raw
        .split(|ch| matches!(ch, '\n' | ',' | ';'))
        .map(str::trim)
        .filter(|value| !value.is_empty())
    {
        let separator_index = entry.find(['|', ':', '=']);
        let Some(index) = separator_index else {
            continue;
        };
        let email = normalize_email_address(entry[..index].trim());
        let role = entry[index + 1..].trim();
        if email.is_empty() || role.is_empty() {
            continue;
        }
        roles.insert(email, role.to_string());
    }
    roles
}

fn founder_email_role_summaries(settings: &BTreeMap<String, String>) -> Vec<String> {
    let roles = parse_founder_email_roles(settings);
    parse_founder_email_addresses(settings)
        .into_iter()
        .map(|email| {
            let role = roles
                .get(&email)
                .cloned()
                .unwrap_or_else(|| "Founder".to_string());
            format!("{email} ({role})")
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
    value
        .trim()
        .trim_matches('<')
        .trim_matches('>')
        .to_lowercase()
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

pub(crate) fn refresh_thread(conn: &mut Connection, thread_key: &str) -> Result<()> {
    let tx = conn.unchecked_transaction()?;
    refresh_thread_tx(&tx, thread_key)?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn upsert_communication_account(
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

pub(crate) fn record_communication_sync_run(
    conn: &mut Connection,
    run: CommunicationSyncRun<'_>,
) -> Result<()> {
    if run.ok && run.stored_count <= 0 && run.error_text.trim().is_empty() {
        return Ok(());
    }
    conn.execute(
        r#"
        INSERT INTO communication_sync_runs (
            run_key, channel, account_key, folder_hint, started_at, finished_at,
            ok, fetched_count, stored_count, error_text, metadata_json
        ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
        "#,
        params![
            run.run_key,
            run.channel,
            run.account_key,
            run.folder_hint,
            run.started_at,
            run.finished_at,
            if run.ok { 1 } else { 0 },
            run.fetched_count,
            run.stored_count,
            run.error_text,
            run.metadata_json,
        ],
    )?;
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

pub(crate) fn preview_text(body: &str, subject: &str) -> String {
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

pub(crate) fn stable_digest(input: &str) -> String {
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

fn jami_address_from_account_key(account_key: &str) -> String {
    account_key
        .strip_prefix("jami:")
        .unwrap_or(account_key)
        .to_string()
}

fn teams_tenant_from_account_config(account_config: Option<&AccountConfig>) -> Option<String> {
    account_config
        .and_then(|config| config.profile_json.get("tenantId"))
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|tenant_id| !tenant_id.is_empty())
        .map(ToOwned::to_owned)
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
        .unwrap_or_else(|| crate::paths::core_db(&root))
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

fn print_json(value: &Value) -> Result<()> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

pub(crate) fn now_iso_string() -> String {
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
mod tests;

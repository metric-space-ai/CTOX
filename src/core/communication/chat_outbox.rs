// Origin: CTOX
// License: AGPL-3.0-only

use anyhow::{Context, Result};
use rusqlite::{params, Connection, OptionalExtension, Transaction};
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

use crate::communication_store::{now_iso_string, open_channel_db, refresh_thread, UpsertMessage};

const OUTBOX_MAX_ATTEMPTS: u32 = 5;
const OUTBOX_BATCH_LIMIT: usize = 100;
const OUTBOX_CLAIM_LEASE_MS: i64 = 60_000;
const OUTBOX_DUPLICATE_POLL_TTL: Duration = Duration::from_secs(30);

#[derive(Clone, Copy, Debug)]
struct OutboxPollGateEntry {
    last_started: Option<Instant>,
    generation: u64,
    processed_generation: u64,
}

static OUTBOX_POLL_GATE: OnceLock<Mutex<BTreeMap<PathBuf, OutboxPollGateEntry>>> = OnceLock::new();

#[derive(Clone, Debug)]
pub(crate) struct OutboxError {
    pub kind: String,
    pub message: String,
    pub http_status: Option<u16>,
    pub retry_after_seconds: Option<i64>,
}

impl OutboxError {
    pub(crate) fn provider(message: impl Into<String>) -> Self {
        Self {
            kind: "provider_error".to_string(),
            message: message.into(),
            http_status: None,
            retry_after_seconds: None,
        }
    }

    pub(crate) fn typed_http(
        message: impl Into<String>,
        status: u16,
        retry_after_seconds: Option<i64>,
    ) -> Self {
        Self {
            kind: "chat_http_error".to_string(),
            message: message.into(),
            http_status: Some(status),
            retry_after_seconds,
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({
            "type": self.kind,
            "message": self.message,
            "status": self.http_status,
            "http_status": self.http_status,
            "retry_after_seconds": self.retry_after_seconds,
        })
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NewOutboxItem {
    pub local_id: String,
    pub message_key: String,
    pub channel: String,
    pub account_key: String,
    pub destination: String,
    pub thread_key: String,
    pub sender_display: String,
    pub sender_address: String,
    pub recipient_addresses: Vec<String>,
    pub cc_addresses: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub attachment_paths: Vec<String>,
    pub provider_metadata: Value,
}

#[derive(Clone, Debug)]
pub(crate) struct OutboxItem {
    pub local_id: String,
    pub message_key: String,
    pub channel: String,
    pub account_key: String,
    pub destination: String,
    pub thread_key: String,
    pub sender_display: String,
    pub sender_address: String,
    pub recipient_addresses: Vec<String>,
    pub cc_addresses: Vec<String>,
    pub subject: String,
    pub body_text: String,
    pub attachment_paths: Vec<String>,
    pub provider_metadata: Value,
    pub attempt_count: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct OutboxSendSuccess {
    pub remote_id: String,
    pub provider_response: Value,
    pub metadata_patch: Value,
}

impl OutboxSendSuccess {
    pub(crate) fn new(remote_id: String, provider_response: Value) -> Self {
        Self {
            remote_id,
            provider_response,
            metadata_patch: json!({}),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct OutboxSchedule {
    pub attempt_count: u32,
    pub next_attempt_at_ms: i64,
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct OutboxProcessSummary {
    pub attempted: usize,
    pub sent: usize,
    pub requeued: usize,
    pub failed_permanent: usize,
}

impl OutboxProcessSummary {
    fn to_json(self) -> Value {
        json!({
            "ok": true,
            "attempted": self.attempted,
            "sent": self.sent,
            "requeued": self.requeued,
            "failed_permanent": self.failed_permanent,
        })
    }
}

pub(crate) fn new_local_id() -> String {
    // UUIDv4 is independent of destination, body, and wall-clock resolution, so
    // two identical sends cannot overwrite each other in the local outbox.
    format!("queued-{}", uuid::Uuid::new_v4())
}

pub(crate) fn current_unix_millis() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

pub(crate) fn persist_queued_message(
    conn: &mut Connection,
    message: UpsertMessage<'_>,
    outbox: &NewOutboxItem,
    error: &OutboxError,
    now_ms: i64,
) -> Result<OutboxSchedule> {
    ensure_outbox_schema(conn)?;
    let attempt_count = 1;
    let next_attempt_at_ms = next_attempt_at_ms(now_ms, attempt_count, error);
    let now = now_iso_string();
    let recipients_json = serde_json::to_string(&outbox.recipient_addresses)?;
    let cc_json = serde_json::to_string(&outbox.cc_addresses)?;
    let attachments_json = serde_json::to_string(&outbox.attachment_paths)?;
    let provider_metadata_json = serde_json::to_string(&outbox.provider_metadata)?;
    let error_json = serde_json::to_string(&error.to_json())?;

    let tx = conn.transaction()?;
    upsert_message_tx(&tx, message)?;
    tx.execute(
        r#"
        INSERT INTO communication_chat_outbox (
            local_id, message_key, channel, account_key, destination, thread_key,
            sender_display, sender_address, recipient_addresses_json, cc_addresses_json,
            subject, body_text, attachment_paths_json, provider_metadata_json,
            status, attempt_count, next_attempt_at, last_error_json, remote_id,
            created_at, updated_at
        ) VALUES (
            ?1, ?2, ?3, ?4, ?5, ?6,
            ?7, ?8, ?9, ?10,
            ?11, ?12, ?13, ?14,
            'queued', ?15, ?16, ?17, NULL,
            ?18, ?18
        )
        ON CONFLICT(local_id) DO UPDATE SET
            message_key=excluded.message_key,
            channel=excluded.channel,
            account_key=excluded.account_key,
            destination=excluded.destination,
            thread_key=excluded.thread_key,
            sender_display=excluded.sender_display,
            sender_address=excluded.sender_address,
            recipient_addresses_json=excluded.recipient_addresses_json,
            cc_addresses_json=excluded.cc_addresses_json,
            subject=excluded.subject,
            body_text=excluded.body_text,
            attachment_paths_json=excluded.attachment_paths_json,
            provider_metadata_json=excluded.provider_metadata_json,
            status='queued',
            attempt_count=excluded.attempt_count,
            next_attempt_at=excluded.next_attempt_at,
            last_error_json=excluded.last_error_json,
            remote_id=NULL,
            updated_at=excluded.updated_at
        "#,
        params![
            outbox.local_id,
            outbox.message_key,
            outbox.channel,
            outbox.account_key,
            outbox.destination,
            outbox.thread_key,
            outbox.sender_display,
            outbox.sender_address,
            recipients_json,
            cc_json,
            outbox.subject,
            outbox.body_text,
            attachments_json,
            provider_metadata_json,
            attempt_count,
            next_attempt_at_ms,
            error_json,
            now,
        ],
    )?;
    tx.commit()?;
    mark_outbox_poll_dirty(conn);

    Ok(OutboxSchedule {
        attempt_count,
        next_attempt_at_ms,
    })
}

pub(crate) fn process_chat_outbox(root: &Path) -> Result<Value> {
    let db_path = root.join("runtime/ctox.sqlite3");
    if !claim_outbox_poll(&db_path, Instant::now()) {
        return Ok(OutboxProcessSummary::default().to_json());
    }
    let summary = process_chat_outbox_for_channel_with_sender(
        &db_path,
        None,
        current_unix_millis(),
        |item| {
            if item.channel == "teams" {
                crate::communication::teams_native::send_outbox_item(root, item)
            } else {
                crate::communication::chat_native::send_outbox_item(root, item)
            }
        },
    )?;
    Ok(summary.to_json())
}

fn claim_outbox_poll(db_path: &Path, now: Instant) -> bool {
    let gate = OUTBOX_POLL_GATE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut entries = gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = entries
        .entry(outbox_poll_key(db_path))
        .or_insert(OutboxPollGateEntry {
            last_started: None,
            generation: 0,
            processed_generation: 0,
        });
    let unchanged_since_last_poll = entry.generation == entry.processed_generation;
    let inside_ttl = entry
        .last_started
        .map(|last_started| now.saturating_duration_since(last_started) < OUTBOX_DUPLICATE_POLL_TTL)
        .unwrap_or(false);
    if unchanged_since_last_poll && inside_ttl {
        return false;
    }
    entry.last_started = Some(now);
    entry.processed_generation = entry.generation;
    true
}

fn mark_outbox_poll_dirty(conn: &Connection) {
    let Some(path) = conn.path() else {
        return;
    };
    let gate = OUTBOX_POLL_GATE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let mut entries = gate.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let entry = entries
        .entry(outbox_poll_key(Path::new(path)))
        .or_insert(OutboxPollGateEntry {
            last_started: None,
            generation: 0,
            processed_generation: 0,
        });
    entry.generation = entry.generation.wrapping_add(1);
}

fn outbox_poll_key(db_path: &Path) -> PathBuf {
    if let Ok(path) = db_path.canonicalize() {
        return path;
    }
    let Some(file_name) = db_path.file_name() else {
        return db_path.to_path_buf();
    };
    db_path
        .parent()
        .and_then(|parent| parent.canonicalize().ok())
        .map(|parent| parent.join(file_name))
        .unwrap_or_else(|| db_path.to_path_buf())
}

pub(crate) fn process_chat_outbox_for_channel_with_sender<F>(
    db_path: &Path,
    channel: Option<&str>,
    now_ms: i64,
    mut sender: F,
) -> Result<OutboxProcessSummary>
where
    F: FnMut(&OutboxItem) -> std::result::Result<OutboxSendSuccess, OutboxError>,
{
    let mut conn = open_channel_db(db_path)?;
    ensure_outbox_schema(&conn)?;
    let due_ids = due_local_ids(&conn, channel, now_ms)?;
    let mut summary = OutboxProcessSummary::default();

    for local_id in due_ids {
        if !claim_item(&conn, &local_id, now_ms)? {
            continue;
        }
        let Some(item) = load_item(&conn, &local_id)? else {
            continue;
        };
        summary.attempted += 1;
        match sender(&item) {
            Ok(success) => {
                mark_sent(&mut conn, &item, success)?;
                summary.sent += 1;
            }
            Err(error) => {
                if mark_failed_attempt(&mut conn, &item, &error, now_ms)? {
                    summary.failed_permanent += 1;
                } else {
                    summary.requeued += 1;
                }
            }
        }
    }

    Ok(summary)
}

fn ensure_outbox_schema(conn: &Connection) -> Result<()> {
    conn.execute_batch(
        r#"
        CREATE TABLE IF NOT EXISTS communication_chat_outbox (
            local_id TEXT PRIMARY KEY,
            message_key TEXT NOT NULL UNIQUE,
            channel TEXT NOT NULL,
            account_key TEXT NOT NULL,
            destination TEXT NOT NULL,
            thread_key TEXT NOT NULL,
            sender_display TEXT NOT NULL,
            sender_address TEXT NOT NULL,
            recipient_addresses_json TEXT NOT NULL,
            cc_addresses_json TEXT NOT NULL,
            subject TEXT NOT NULL,
            body_text TEXT NOT NULL,
            attachment_paths_json TEXT NOT NULL DEFAULT '[]',
            provider_metadata_json TEXT NOT NULL DEFAULT '{}',
            status TEXT NOT NULL DEFAULT 'queued',
            attempt_count INTEGER NOT NULL DEFAULT 0,
            next_attempt_at INTEGER,
            last_error_json TEXT NOT NULL DEFAULT '{}',
            remote_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        "#,
    )?;

    // Keep upgrades idempotent for databases that briefly carried an earlier
    // outbox draft. SQLite has no ADD COLUMN IF NOT EXISTS, hence PRAGMA guard.
    ensure_column(
        conn,
        "status",
        "ALTER TABLE communication_chat_outbox ADD COLUMN status TEXT NOT NULL DEFAULT 'queued'",
    )?;
    ensure_column(
        conn,
        "attempt_count",
        "ALTER TABLE communication_chat_outbox ADD COLUMN attempt_count INTEGER NOT NULL DEFAULT 0",
    )?;
    ensure_column(
        conn,
        "next_attempt_at",
        "ALTER TABLE communication_chat_outbox ADD COLUMN next_attempt_at INTEGER",
    )?;
    ensure_column(
        conn,
        "last_error_json",
        "ALTER TABLE communication_chat_outbox ADD COLUMN last_error_json TEXT NOT NULL DEFAULT '{}'",
    )?;
    ensure_column(
        conn,
        "remote_id",
        "ALTER TABLE communication_chat_outbox ADD COLUMN remote_id TEXT",
    )?;
    conn.execute_batch(
        r#"
        CREATE INDEX IF NOT EXISTS idx_communication_chat_outbox_due
            ON communication_chat_outbox(status, next_attempt_at, channel);
        "#,
    )?;
    Ok(())
}

fn ensure_column(conn: &Connection, column: &str, migration: &str) -> Result<()> {
    let mut stmt = conn.prepare("PRAGMA table_info(communication_chat_outbox)")?;
    let names = stmt.query_map([], |row| row.get::<_, String>(1))?;
    for name in names {
        if name? == column {
            return Ok(());
        }
    }
    conn.execute_batch(migration)?;
    Ok(())
}

fn due_local_ids(conn: &Connection, channel: Option<&str>, now_ms: i64) -> Result<Vec<String>> {
    let mut stmt = conn.prepare(
        r#"
        SELECT local_id
        FROM communication_chat_outbox
        WHERE status = 'queued'
          AND next_attempt_at IS NOT NULL
          AND next_attempt_at <= ?1
          AND (?2 IS NULL OR channel = ?2)
        ORDER BY next_attempt_at ASC, created_at ASC
        LIMIT ?3
        "#,
    )?;
    let rows = stmt.query_map(params![now_ms, channel, OUTBOX_BATCH_LIMIT as i64], |row| {
        row.get::<_, String>(0)
    })?;
    rows.collect::<std::result::Result<Vec<_>, _>>()
        .map_err(Into::into)
}

fn claim_item(conn: &Connection, local_id: &str, now_ms: i64) -> Result<bool> {
    let claimed = conn.execute(
        r#"
        UPDATE communication_chat_outbox
        SET next_attempt_at = ?2,
            updated_at = ?3
        WHERE local_id = ?1
          AND status = 'queued'
          AND next_attempt_at IS NOT NULL
          AND next_attempt_at <= ?4
        "#,
        params![
            local_id,
            now_ms.saturating_add(OUTBOX_CLAIM_LEASE_MS),
            now_iso_string(),
            now_ms,
        ],
    )?;
    Ok(claimed == 1)
}

fn load_item(conn: &Connection, local_id: &str) -> Result<Option<OutboxItem>> {
    conn.query_row(
        r#"
        SELECT local_id, message_key, channel, account_key, destination, thread_key,
               sender_display, sender_address, recipient_addresses_json, cc_addresses_json,
               subject, body_text, attachment_paths_json, provider_metadata_json, attempt_count
        FROM communication_chat_outbox
        WHERE local_id = ?1 AND status = 'queued'
        "#,
        params![local_id],
        |row| {
            let recipients_json: String = row.get(8)?;
            let cc_json: String = row.get(9)?;
            let attachments_json: String = row.get(12)?;
            let provider_metadata_json: String = row.get(13)?;
            let attempt_count: i64 = row.get(14)?;
            Ok(OutboxItem {
                local_id: row.get(0)?,
                message_key: row.get(1)?,
                channel: row.get(2)?,
                account_key: row.get(3)?,
                destination: row.get(4)?,
                thread_key: row.get(5)?,
                sender_display: row.get(6)?,
                sender_address: row.get(7)?,
                recipient_addresses: serde_json::from_str(&recipients_json).unwrap_or_default(),
                cc_addresses: serde_json::from_str(&cc_json).unwrap_or_default(),
                subject: row.get(10)?,
                body_text: row.get(11)?,
                attachment_paths: serde_json::from_str(&attachments_json).unwrap_or_default(),
                provider_metadata: serde_json::from_str(&provider_metadata_json)
                    .unwrap_or_else(|_| json!({})),
                attempt_count: attempt_count.max(0).min(u32::MAX as i64) as u32,
            })
        },
    )
    .optional()
    .map_err(Into::into)
}

fn mark_sent(conn: &mut Connection, item: &OutboxItem, success: OutboxSendSuccess) -> Result<()> {
    let attempt_count = item.attempt_count.saturating_add(1);
    let new_message_key = format!("{}::SENT::{}", item.account_key, success.remote_id);
    let generated_thread = item
        .provider_metadata
        .get("generatedThreadKey")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let final_thread_key = if generated_thread {
        item.thread_key.replace(&item.local_id, &success.remote_id)
    } else {
        item.thread_key.clone()
    };
    let now = now_iso_string();
    let metadata_json = updated_message_metadata(
        conn,
        &item.message_key,
        "sent",
        attempt_count,
        None,
        Some(&success.provider_response),
        Some(&success.metadata_patch),
    )?;

    let tx = conn.transaction()?;
    tx.execute(
        "DELETE FROM communication_messages WHERE message_key = ?1 AND message_key <> ?2",
        params![new_message_key, item.message_key],
    )?;
    tx.execute(
        r#"
        UPDATE communication_messages
        SET message_key = ?1,
            remote_id = ?2,
            thread_key = ?3,
            status = 'sent',
            observed_at = ?4,
            metadata_json = ?5
        WHERE message_key = ?6
        "#,
        params![
            new_message_key,
            success.remote_id,
            final_thread_key,
            now,
            metadata_json,
            item.message_key,
        ],
    )?;
    tx.execute(
        r#"
        UPDATE communication_chat_outbox
        SET message_key = ?1,
            thread_key = ?2,
            status = 'sent',
            attempt_count = ?3,
            next_attempt_at = NULL,
            last_error_json = '{}',
            remote_id = ?4,
            updated_at = ?5
        WHERE local_id = ?6
        "#,
        params![
            new_message_key,
            final_thread_key,
            attempt_count,
            success.remote_id,
            now,
            item.local_id,
        ],
    )?;
    tx.execute(
        r#"
        UPDATE communication_accounts
        SET last_outbound_ok_at = ?2, updated_at = ?2
        WHERE account_key = ?1
        "#,
        params![item.account_key, now],
    )?;
    tx.commit()?;

    refresh_thread(conn, &item.thread_key)?;
    if final_thread_key != item.thread_key {
        refresh_thread(conn, &final_thread_key)?;
    }
    Ok(())
}

fn mark_failed_attempt(
    conn: &mut Connection,
    item: &OutboxItem,
    error: &OutboxError,
    now_ms: i64,
) -> Result<bool> {
    let attempt_count = item.attempt_count.saturating_add(1);
    let permanent = attempt_count >= OUTBOX_MAX_ATTEMPTS;
    let status = if permanent {
        "failed_permanent"
    } else {
        "queued"
    };
    let next_attempt = (!permanent).then(|| next_attempt_at_ms(now_ms, attempt_count, error));
    let now = now_iso_string();
    let error_json = serde_json::to_string(&error.to_json())?;
    let metadata_json = updated_message_metadata(
        conn,
        &item.message_key,
        status,
        attempt_count,
        Some(&error.to_json()),
        None,
        None,
    )?;

    let tx = conn.transaction()?;
    tx.execute(
        r#"
        UPDATE communication_messages
        SET status = ?1,
            observed_at = ?2,
            metadata_json = ?3
        WHERE message_key = ?4
        "#,
        params![status, now, metadata_json, item.message_key],
    )?;
    tx.execute(
        r#"
        UPDATE communication_chat_outbox
        SET status = ?1,
            attempt_count = ?2,
            next_attempt_at = ?3,
            last_error_json = ?4,
            updated_at = ?5
        WHERE local_id = ?6
        "#,
        params![
            status,
            attempt_count,
            next_attempt,
            error_json,
            now,
            item.local_id,
        ],
    )?;
    tx.commit()?;
    refresh_thread(conn, &item.thread_key)?;
    Ok(permanent)
}

fn updated_message_metadata(
    conn: &Connection,
    message_key: &str,
    status: &str,
    attempt_count: u32,
    error: Option<&Value>,
    provider_response: Option<&Value>,
    metadata_patch: Option<&Value>,
) -> Result<String> {
    let raw = conn
        .query_row(
            "SELECT metadata_json FROM communication_messages WHERE message_key = ?1",
            params![message_key],
            |row| row.get::<_, String>(0),
        )
        .optional()?
        .unwrap_or_else(|| "{}".to_string());
    let mut metadata = serde_json::from_str::<Value>(&raw).unwrap_or_else(|_| json!({}));
    if !metadata.is_object() {
        metadata = json!({"previousMetadata": metadata});
    }
    let object = metadata
        .as_object_mut()
        .expect("metadata normalized to object");
    object.insert(
        "outbox".to_string(),
        json!({
            "status": status,
            "attempt_count": attempt_count,
            "last_error": error.cloned().unwrap_or(Value::Null),
        }),
    );
    if let Some(response) = provider_response {
        object.insert("providerResponse".to_string(), response.clone());
        object.insert("error".to_string(), Value::Null);
    }
    if let Some(error) = error {
        object.insert("error".to_string(), error.clone());
    }
    if let Some(Value::Object(patch)) = metadata_patch {
        merge_object(object, patch);
    }
    Ok(serde_json::to_string(&metadata)?)
}

fn merge_object(target: &mut Map<String, Value>, patch: &Map<String, Value>) {
    for (key, value) in patch {
        target.insert(key.clone(), value.clone());
    }
}

fn next_attempt_at_ms(now_ms: i64, attempt_count: u32, error: &OutboxError) -> i64 {
    let backoff_ms = outbox_backoff_delay_ms(attempt_count);
    let retry_after_ms = error
        .retry_after_seconds
        .unwrap_or(0)
        .max(0)
        .saturating_mul(1_000);
    now_ms.saturating_add(backoff_ms.max(retry_after_ms))
}

fn outbox_backoff_delay_ms(attempt_count: u32) -> i64 {
    let shift = attempt_count.saturating_sub(1).min(6);
    (1_000_i64.saturating_mul(1_i64 << shift)).min(60_000)
}

fn upsert_message_tx(tx: &Transaction<'_>, message: UpsertMessage<'_>) -> Result<()> {
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
            message.seen,
            message.has_attachments,
            message.external_created_at,
            message.observed_at,
            message.metadata_json,
        ],
    )
    .context("failed to persist queued communication message")?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::communication_store::{open_channel_db, preview_text};

    #[test]
    fn duplicate_service_polls_are_suppressed_until_dirty_or_ttl() -> Result<()> {
        let root = tempfile::tempdir()?;
        let db_path = root.path().join("runtime/ctox.sqlite3");
        std::fs::create_dir_all(db_path.parent().expect("database parent"))?;
        let started_at = Instant::now();

        assert!(claim_outbox_poll(&db_path, started_at));
        assert!(!claim_outbox_poll(
            &db_path,
            started_at + OUTBOX_DUPLICATE_POLL_TTL - Duration::from_millis(1)
        ));

        let conn = Connection::open(&db_path)?;
        mark_outbox_poll_dirty(&conn);
        assert!(claim_outbox_poll(
            &db_path,
            started_at + Duration::from_secs(1)
        ));
        assert!(!claim_outbox_poll(
            &db_path,
            started_at + Duration::from_secs(2)
        ));
        assert!(claim_outbox_poll(
            &db_path,
            started_at + Duration::from_secs(1) + OUTBOX_DUPLICATE_POLL_TTL
        ));
        Ok(())
    }

    fn queue_test_message(
        conn: &mut Connection,
        channel: &str,
        local_id: &str,
        now_ms: i64,
    ) -> Result<OutboxSchedule> {
        let account_key = format!("{channel}:account");
        let message_key = format!("{account_key}::SENT::{local_id}");
        let thread_key = format!("{account_key}::channel::dest::thread::{local_id}");
        let timestamp = now_iso_string();
        let recipients = vec!["dest".to_string()];
        let recipients_json = serde_json::to_string(&recipients)?;
        let error = OutboxError::typed_http("HTTP 503", 503, None);
        let metadata_json = serde_json::to_string(&json!({"error": error.to_json()}))?;
        let preview = preview_text("same body", "Subject");
        let outbox = NewOutboxItem {
            local_id: local_id.to_string(),
            message_key: message_key.clone(),
            channel: channel.to_string(),
            account_key: account_key.clone(),
            destination: "dest".to_string(),
            thread_key: thread_key.clone(),
            sender_display: "CTOX Bot".to_string(),
            sender_address: "bot@example.test".to_string(),
            recipient_addresses: recipients,
            cc_addresses: Vec::new(),
            subject: "Subject".to_string(),
            body_text: "same body".to_string(),
            attachment_paths: Vec::new(),
            provider_metadata: json!({"generatedThreadKey": true}),
        };
        persist_queued_message(
            conn,
            UpsertMessage {
                message_key: &message_key,
                channel,
                account_key: &account_key,
                thread_key: &thread_key,
                remote_id: local_id,
                direction: "outbound",
                folder_hint: "SENT",
                sender_display: "CTOX Bot",
                sender_address: "bot@example.test",
                recipient_addresses_json: &recipients_json,
                cc_addresses_json: "[]",
                bcc_addresses_json: "[]",
                subject: "Subject",
                preview: &preview,
                body_text: "same body",
                body_html: "",
                raw_payload_ref: "",
                trust_level: "high",
                status: "queued",
                seen: true,
                has_attachments: false,
                external_created_at: &timestamp,
                observed_at: &timestamp,
                metadata_json: &metadata_json,
            },
            &outbox,
            &error,
            now_ms,
        )
    }

    #[test]
    fn retry_after_backoff_sends_and_replaces_local_id() -> Result<()> {
        let root = tempfile::tempdir()?;
        let db_path = root.path().join("runtime/ctox.sqlite3");
        let mut conn = open_channel_db(&db_path)?;
        let local_id = new_local_id();
        let schedule = queue_test_message(&mut conn, "slack", &local_id, 1_000)?;
        drop(conn);

        let mut calls = 0;
        let early = process_chat_outbox_for_channel_with_sender(
            &db_path,
            Some("slack"),
            schedule.next_attempt_at_ms - 1,
            |_| {
                calls += 1;
                Ok(OutboxSendSuccess::new("too-early".to_string(), json!({})))
            },
        )?;
        assert_eq!(early.attempted, 0);
        assert_eq!(calls, 0);

        let summary = process_chat_outbox_for_channel_with_sender(
            &db_path,
            Some("slack"),
            schedule.next_attempt_at_ms,
            |item| {
                calls += 1;
                assert_eq!(item.local_id, local_id);
                Ok(OutboxSendSuccess::new(
                    "1719360000.000099".to_string(),
                    json!({"ok": true, "ts": "1719360000.000099"}),
                ))
            },
        )?;
        assert_eq!(calls, 1);
        assert_eq!(summary.sent, 1);

        let conn = Connection::open(&db_path)?;
        let row: (String, String, String) = conn.query_row(
            "SELECT message_key, remote_id, status FROM communication_messages WHERE direction = 'outbound'",
            [],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?)),
        )?;
        assert_eq!(row.0, "slack:account::SENT::1719360000.000099");
        assert_eq!(row.1, "1719360000.000099");
        assert_eq!(row.2, "sent");
        let outbox_status: String = conn.query_row(
            "SELECT status FROM communication_chat_outbox WHERE local_id = ?1",
            params![local_id],
            |row| row.get(0),
        )?;
        assert_eq!(outbox_status, "sent");
        Ok(())
    }

    #[test]
    fn exhausted_budget_is_permanent_and_never_retried() -> Result<()> {
        let root = tempfile::tempdir()?;
        let db_path = root.path().join("runtime/ctox.sqlite3");
        let mut conn = open_channel_db(&db_path)?;
        let local_id = new_local_id();
        let schedule = queue_test_message(&mut conn, "teams", &local_id, 1_000)?;
        conn.execute(
            "UPDATE communication_chat_outbox SET attempt_count = ?1 WHERE local_id = ?2",
            params![OUTBOX_MAX_ATTEMPTS - 1, local_id],
        )?;
        drop(conn);

        let mut calls = 0;
        let summary = process_chat_outbox_for_channel_with_sender(
            &db_path,
            Some("teams"),
            schedule.next_attempt_at_ms,
            |_| {
                calls += 1;
                Err(OutboxError::provider("still unavailable"))
            },
        )?;
        assert_eq!(calls, 1);
        assert_eq!(summary.failed_permanent, 1);

        let summary =
            process_chat_outbox_for_channel_with_sender(&db_path, Some("teams"), i64::MAX, |_| {
                calls += 1;
                Ok(OutboxSendSuccess::new(
                    "must-not-send".to_string(),
                    json!({}),
                ))
            })?;
        assert_eq!(summary.attempted, 0);
        assert_eq!(calls, 1);

        let conn = Connection::open(&db_path)?;
        let row: (String, Option<i64>) = conn.query_row(
            "SELECT status, next_attempt_at FROM communication_chat_outbox WHERE local_id = ?1",
            params![local_id],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(row.0, "failed_permanent");
        assert_eq!(row.1, None);
        let message_status: String = conn.query_row(
            "SELECT status FROM communication_messages WHERE remote_id = ?1",
            params![local_id],
            |row| row.get(0),
        )?;
        assert_eq!(message_status, "failed_permanent");
        Ok(())
    }

    #[test]
    fn migration_adds_retry_columns_once() -> Result<()> {
        let conn = Connection::open_in_memory()?;
        conn.execute_batch(
            r#"
            CREATE TABLE communication_chat_outbox (
                local_id TEXT PRIMARY KEY,
                message_key TEXT NOT NULL UNIQUE,
                channel TEXT NOT NULL,
                account_key TEXT NOT NULL,
                destination TEXT NOT NULL,
                thread_key TEXT NOT NULL,
                sender_display TEXT NOT NULL,
                sender_address TEXT NOT NULL,
                recipient_addresses_json TEXT NOT NULL,
                cc_addresses_json TEXT NOT NULL,
                subject TEXT NOT NULL,
                body_text TEXT NOT NULL,
                attachment_paths_json TEXT NOT NULL DEFAULT '[]',
                provider_metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );
            "#,
        )?;
        ensure_outbox_schema(&conn)?;
        ensure_outbox_schema(&conn)?;
        let mut stmt = conn.prepare("PRAGMA table_info(communication_chat_outbox)")?;
        let columns = stmt
            .query_map([], |row| row.get::<_, String>(1))?
            .collect::<std::result::Result<Vec<_>, _>>()?;
        for required in [
            "status",
            "attempt_count",
            "next_attempt_at",
            "last_error_json",
            "remote_id",
        ] {
            assert!(columns.iter().any(|column| column == required));
        }
        Ok(())
    }
}

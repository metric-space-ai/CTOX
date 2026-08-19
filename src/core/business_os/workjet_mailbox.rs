// Origin: CTOX
// License: AGPL-3.0-only

//! Workjet mailbox envelopes: a synced, CTOX-opaque envelope collection.
//!
//! The Workjet Code server on each machine needs a way to hand a signed routing
//! envelope plus an opaque payload to the Workjet Code server on another
//! machine that shares the same CTOX sync room. CTOX already replicates a
//! Business OS RxDB room between its instances, so the docking decision is:
//! **the daemon replicates, Workjet keeps every mailbox semantic.** CTOX never
//! parses, verifies or interprets `envelope_json` / `payload_json`; it only
//! enforces bounds, charset and the wire budget that keeps replication alive.
//!
//! Why a dedicated collection instead of reusing `business_commands`: the
//! command collection carries CTOX-owned semantics (status machine, client
//! context redaction, per-collection authz defaults). Mailbox envelopes are
//! foreign payloads with a foreign lifecycle; mixing them would make every
//! CTOX command consumer defensive about documents it must never touch.
//!
//! Registration deliberately does NOT go through
//! `business_os_schema_contract.json`: that file is GENERATED from
//! `src/apps/business-os/modules/*/schema.js` by
//! `src/core/rxdb/tools/build_business_os_schema_contract.mjs`, and CI
//! re-runs the generator (`.github/workflows/ci.yml`) to detect drift. A
//! daemon-owned collection that no browser shell module declares is therefore
//! injected into the creator map instead, from
//! `rxdb_peer::collection_creators`, which is the single map that both
//! `add_collections` and `add_collections_tolerant` consume — so the mailbox
//! joins the one multiplexed WebRTC replication session for the whole room
//! exactly like every contract collection, and
//! `store::ensure_legacy_collection_grants` materializes its sync grants from
//! the same registered-collection list.

use std::collections::HashMap;
use std::collections::HashSet;
use std::path::Path;
use std::time::Duration;
use std::time::SystemTime;
use std::time::UNIX_EPOCH;

use anyhow::Context;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use base64::Engine as _;
use rusqlite::params;
use rusqlite::Connection;
use rusqlite::OptionalExtension;
use rxdb::rx_database::RxCollectionCreator;
use rxdb::types::RxJsonSchema;
use serde_json::json;
use serde_json::Value;
use uuid::Uuid;

/// The replicated collection name. Lowercase + underscores only, so it passes
/// `store::is_safe_rxdb_collection_name` and the RxDB table-name contract.
pub(super) const MAILBOX_COLLECTION: &str = "workjet_mailbox_envelopes";

/// Schema version 0. `store::rxdb_schema_version` returns 0 for every
/// collection that is not in the generated contract, so version 0 makes the
/// table name the peer creates (`ctox_business_os__…__v0`) and the table name
/// the store resolves agree without a second source of truth.
const MAILBOX_SCHEMA_VERSION: i64 = 0;

pub(super) const MAILBOX_TABLE: &str = "ctox_business_os__workjet_mailbox_envelopes__v0";

/// Envelope ids and routing ids are bounded and charset-restricted. CTOX
/// validates nothing else about them: they are Workjet identifiers.
pub(super) const MAX_ID_BYTES: usize = 128;

/// The signed Workjet routing envelope, opaque to CTOX.
pub(super) const MAX_ENVELOPE_JSON_BYTES: usize = 8 * 1024;

/// The opaque payload blob.
///
/// The brief asked for 8 MiB. That ceiling cannot hold: an eagerly replicated
/// document travels as ONE query-fetch document, and
/// `rxdb_peer::retain_projectable_knowledge_item` documents the consequence of
/// exceeding the wire chunk budget — the browser's initial replication for the
/// WHOLE collection stalls silently, `initialReplicationAt` stays null and no
/// error is raised. That guard pins the budget at 262_144 bytes, which matches
/// `CTOX_QUERY_MAX_BYTES_PER_CHUNK` in the generated replication protocol
/// contract. The mailbox therefore uses a documented LOWER ceiling: 200_000
/// bytes of payload, with the whole serialized document held under the
/// 262_144-byte wire budget. Anything larger belongs in a demand-chunked
/// collection, not here.
pub(super) const MAX_PAYLOAD_JSON_BYTES: usize = 200_000;

/// Hard wire budget for the complete serialized envelope document.
pub(super) const MAX_DOCUMENT_BYTES: usize = 262_144;

/// `consumed_by` grows only by union; bound it so a misbehaving room cannot
/// grow one document without limit.
pub(super) const MAX_CONSUMED_BY: usize = 64;

const DEFAULT_PAGE_LIMIT: usize = 50;
const MAX_PAGE_LIMIT: usize = 200;

/// Sweep cadence and batch. Bounded batch + log-and-continue: an unreadable
/// row must never stop the daemon's background maintenance.
const SWEEP_INTERVAL_SECS: u64 = 300;
pub(super) const SWEEP_BATCH_LIMIT: usize = 200;

/// Default envelope lifetime when the publisher does not set `expires_at_ms`.
const DEFAULT_TTL_MS: i64 = 24 * 60 * 60 * 1000;

pub(super) fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|elapsed| elapsed.as_millis().min(i64::MAX as u128) as i64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Collection registration
// ---------------------------------------------------------------------------

/// Injects the mailbox creator into the Business OS collection creator map.
///
/// Called from `rxdb_peer::collection_creators`, the single place both the
/// strict and the tolerant registration path build their map. `or_insert_with`
/// keeps a shell-declared collection of the same name authoritative if one is
/// ever added, so this can never silently shadow the generated contract.
pub(super) fn with_mailbox_collection(
    mut creators: HashMap<String, RxCollectionCreator>,
) -> HashMap<String, RxCollectionCreator> {
    creators
        .entry(MAILBOX_COLLECTION.to_string())
        .or_insert_with(|| RxCollectionCreator {
            schema: mailbox_rx_schema(),
            conflict_handler: None,
            options: HashMap::new(),
        });
    creators
}

/// The mailbox JSON schema.
///
/// No `indexes` are declared on purpose. `add_collections_tolerant` SKIPS a
/// collection whose schema fails validation and only logs it, so an index that
/// rxdb-rs rejects would turn into a silently missing mailbox. The SQLite
/// storage still indexes `(lastWriteTime, id)` and the primary key, and every
/// mailbox query below is bounded by an explicit `LIMIT`.
pub(super) fn mailbox_schema_json() -> Value {
    json!({
        "version": MAILBOX_SCHEMA_VERSION,
        "primaryKey": "id",
        "type": "object",
        "properties": {
            "id": { "type": "string", "maxLength": MAX_ID_BYTES },
            "source_workspace_id": { "type": "string", "maxLength": MAX_ID_BYTES },
            "target_workspace_id": { "type": "string", "maxLength": MAX_ID_BYTES },
            "source_environment_id": { "type": "string", "maxLength": MAX_ID_BYTES },
            "target_environment_id": { "type": "string", "maxLength": MAX_ID_BYTES },
            "created_at_ms": { "type": "number" },
            "expires_at_ms": { "type": "number" },
            "updated_at_ms": { "type": "number" },
            "envelope_json": { "type": "string" },
            "payload_json": { "type": "string" },
            "consumed_by": { "type": "array", "items": { "type": "string" } }
        },
        "required": ["id", "target_environment_id", "envelope_json"],
        "additionalProperties": true
    })
}

fn mailbox_rx_schema() -> RxJsonSchema {
    serde_json::from_value(mailbox_schema_json())
        .expect("Workjet mailbox RxDB schema must match the rxdb-rs schema type")
}

// ---------------------------------------------------------------------------
// Validation — bounds and charset only, never envelope semantics
// ---------------------------------------------------------------------------

/// `[A-Za-z0-9_-]`, non-empty, at most `MAX_ID_BYTES` bytes.
pub(super) fn is_valid_mailbox_id(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= MAX_ID_BYTES
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn require_id(document: &Value, field: &str) -> anyhow::Result<String> {
    let value = document
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .trim()
        .to_string();
    if !is_valid_mailbox_id(&value) {
        anyhow::bail!("`{field}` must be 1..={MAX_ID_BYTES} bytes of [A-Za-z0-9_-]");
    }
    Ok(value)
}

fn optional_id(document: &Value, field: &str) -> anyhow::Result<Option<String>> {
    let Some(raw) = document.get(field) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    let value = raw
        .as_str()
        .context(format!("`{field}` must be a string when present"))?
        .trim()
        .to_string();
    if value.is_empty() {
        return Ok(None);
    }
    if !is_valid_mailbox_id(&value) {
        anyhow::bail!("`{field}` must be 1..={MAX_ID_BYTES} bytes of [A-Za-z0-9_-]");
    }
    Ok(Some(value))
}

fn bounded_blob(document: &Value, field: &str, max_bytes: usize) -> anyhow::Result<String> {
    let value = document
        .get(field)
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    if value.len() > max_bytes {
        anyhow::bail!(
            "`{field}` is {} bytes, over the {max_bytes} byte ceiling",
            value.len()
        );
    }
    Ok(value)
}

fn optional_epoch_ms(document: &Value, field: &str) -> anyhow::Result<Option<i64>> {
    let Some(raw) = document.get(field) else {
        return Ok(None);
    };
    if raw.is_null() {
        return Ok(None);
    }
    let value = raw
        .as_i64()
        .or_else(|| raw.as_f64().map(|number| number as i64))
        .context(format!("`{field}` must be an epoch-millisecond number"))?;
    if value < 0 {
        anyhow::bail!("`{field}` must not be negative");
    }
    Ok(Some(value))
}

/// Builds the validated envelope document from an untrusted publish request.
///
/// Every check here is a BOUND or a CHARSET check. The envelope signature, the
/// routing semantics and the payload encoding stay entirely Workjet's business.
pub(super) fn build_envelope_document(
    request: &Value,
    now: i64,
) -> anyhow::Result<(String, Value)> {
    if !request.is_object() {
        anyhow::bail!("publish body must be a JSON object");
    }
    let id = require_id(request, "id")?;
    let target_environment_id = require_id(request, "target_environment_id")?;
    let envelope_json = bounded_blob(request, "envelope_json", MAX_ENVELOPE_JSON_BYTES)?;
    if envelope_json.is_empty() {
        anyhow::bail!("`envelope_json` is required");
    }
    let payload_json = bounded_blob(request, "payload_json", MAX_PAYLOAD_JSON_BYTES)?;
    let created_at_ms = optional_epoch_ms(request, "created_at_ms")?.unwrap_or(now);
    let expires_at_ms =
        optional_epoch_ms(request, "expires_at_ms")?.unwrap_or(created_at_ms + DEFAULT_TTL_MS);

    let mut document = json!({
        "id": id,
        "target_environment_id": target_environment_id,
        "created_at_ms": created_at_ms,
        "expires_at_ms": expires_at_ms,
        "updated_at_ms": now,
        "envelope_json": envelope_json,
        "payload_json": payload_json,
        "consumed_by": Value::Array(Vec::new()),
        "is_deleted": false,
        "_deleted": false,
    });
    for field in [
        "source_workspace_id",
        "target_workspace_id",
        "source_environment_id",
    ] {
        if let Some(value) = optional_id(request, field)? {
            document[field] = Value::String(value);
        }
    }
    let encoded = serde_json::to_vec(&document)?;
    if encoded.len() > MAX_DOCUMENT_BYTES {
        anyhow::bail!(
            "envelope document is {} bytes, over the {MAX_DOCUMENT_BYTES} byte replication wire budget",
            encoded.len()
        );
    }
    Ok((id, document))
}

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

/// Opens the native Business OS RxDB store and materializes the mailbox table
/// if it is not there yet.
///
/// Creating the table from the daemon side mirrors
/// `store_catalog_projections::write_module_catalog_projection_to_rxdb`, which
/// does exactly this for `business_module_catalog` in production. It keeps the
/// loopback surface usable before the native peer's first bring-up; the peer's
/// own `CREATE TABLE IF NOT EXISTS` then finds a table of the same shape.
pub(super) fn open_mailbox_store(root: &Path) -> anyhow::Result<Connection> {
    let path = super::store::rxdb_store_path(root);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create RxDB runtime dir {}", parent.display()))?;
    }
    let conn = Connection::open(&path)
        .with_context(|| format!("open native Business OS RxDB store {}", path.display()))?;
    conn.busy_timeout(Duration::from_secs(10))?;
    conn.execute_batch(&format!(
        r#"
        PRAGMA journal_mode = WAL;
        PRAGMA busy_timeout = 10000;
        CREATE TABLE IF NOT EXISTS "{MAILBOX_TABLE}"(
            id TEXT NOT NULL PRIMARY KEY UNIQUE,
            revision TEXT,
            deleted INTEGER NOT NULL CHECK (deleted IN (0, 1)),
            lastWriteTime REAL NOT NULL,
            data TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS "{MAILBOX_TABLE}_lwt_id_idx"
            ON "{MAILBOX_TABLE}"(lastWriteTime, id);
        CREATE INDEX IF NOT EXISTS "{MAILBOX_TABLE}_deleted_lwt_id_idx"
            ON "{MAILBOX_TABLE}"(deleted, lastWriteTime, id);
        "#
    ))?;
    Ok(conn)
}

/// Writes one envelope row in the shape RxDB replication understands.
///
/// # Why the revision is not just a fresh unique string
///
/// This row is replicated. RxDB decides which side of a conflict wins by the
/// HEIGHT of `_rev`, the integer before the first `-` (`plugins::utils::
/// utils_revision`). The loopback writer originally stamped `rev_<uuid>`, which
/// has no parseable height at all, and it kept the revision only in the SQLite
/// column — never inside the document JSON, which is what the storage hands
/// back as the document. The observable consequence was narrow and nasty: a
/// first INSERT still replicated (the remote had nothing to compare against),
/// while every later UPDATE of the same id — the expiry TOMBSTONE above all —
/// was dropped by the receiving peer. An envelope retired on one machine stayed
/// live forever on the other. So the write must (1) carry `_rev` in the
/// document and (2) increment its height over the row's previous revision.
///
/// `_deleted` and `_meta.lwt` are written for the same reason: they are the
/// fields RxDB reads off the document, and `document_columns` in the sqlite
/// storage derives its own columns from exactly these three.
fn write_document(
    conn: &Connection,
    id: &str,
    document: &Value,
    deleted: bool,
    write_time_ms: i64,
) -> anyhow::Result<()> {
    let previous_revision: Option<String> = conn
        .query_row(
            &format!(r#"SELECT revision FROM "{MAILBOX_TABLE}" WHERE id = ?1"#),
            params![id],
            |row| row.get::<_, Option<String>>(0),
        )
        .optional()?
        .flatten();
    let height = previous_revision
        .as_deref()
        .and_then(|revision| revision.split_once('-'))
        .and_then(|(height, _)| height.parse::<u64>().ok())
        .unwrap_or(0)
        .saturating_add(1);
    // `simple()` renders the uuid without dashes, so the revision keeps exactly
    // one `-` and stays parseable by `get_height_of_revision`.
    let revision = format!("{height}-{}", Uuid::new_v4().simple());

    let mut stored = document.clone();
    stored["_rev"] = Value::String(revision.clone());
    stored["_deleted"] = Value::Bool(deleted);
    stored["_meta"] = json!({ "lwt": write_time_ms });
    stored["_attachments"] = json!({});

    conn.execute(
        &format!(
            r#"
            INSERT INTO "{MAILBOX_TABLE}" (id, revision, deleted, lastWriteTime, data)
            VALUES (?1, ?2, ?3, ?4, ?5)
            ON CONFLICT(id) DO UPDATE SET
                revision = excluded.revision,
                deleted = excluded.deleted,
                lastWriteTime = excluded.lastWriteTime,
                data = excluded.data
            "#
        ),
        params![
            id,
            revision,
            i64::from(deleted),
            write_time_ms as f64,
            serde_json::to_string(&stored)?
        ],
    )?;
    Ok(())
}

fn load_document(conn: &Connection, id: &str) -> anyhow::Result<Option<(bool, Value)>> {
    let row = conn
        .query_row(
            &format!(r#"SELECT deleted, data FROM "{MAILBOX_TABLE}" WHERE id = ?1"#),
            params![id],
            |row| Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?)),
        )
        .optional()?;
    let Some((deleted, raw)) = row else {
        return Ok(None);
    };
    Ok(Some((deleted != 0, serde_json::from_str(&raw)?)))
}

// ---------------------------------------------------------------------------
// Intake / outtake operations
// ---------------------------------------------------------------------------

/// Inserts one envelope. Idempotent: a repeated id is reported as a duplicate
/// and the stored document is left untouched (including its `consumed_by`
/// progress, which a re-publish must never reset).
pub(super) fn publish_envelope(root: &Path, request: &Value) -> anyhow::Result<Value> {
    let now = now_ms();
    let (id, document) = build_envelope_document(request, now)?;
    let conn = open_mailbox_store(root)?;
    if let Some((deleted, _existing)) = load_document(&conn, &id)? {
        return Ok(json!({
            "ok": true,
            "id": id,
            "duplicate": true,
            "tombstoned": deleted,
        }));
    }
    write_document(&conn, &id, &document, false, now)?;
    Ok(json!({
        "ok": true,
        "id": id,
        "duplicate": false,
        "tombstoned": false,
    }))
}

fn encode_cursor(created_at_ms: i64, id: &str) -> String {
    URL_SAFE_NO_PAD.encode(format!("{created_at_ms}:{id}"))
}

fn decode_cursor(cursor: &str) -> anyhow::Result<(i64, String)> {
    let raw = URL_SAFE_NO_PAD
        .decode(cursor.as_bytes())
        .context("`after` is not a valid mailbox cursor")?;
    let decoded = String::from_utf8(raw).context("`after` is not a valid mailbox cursor")?;
    let (created, id) = decoded
        .split_once(':')
        .context("`after` is not a valid mailbox cursor")?;
    let created_at_ms: i64 = created
        .parse()
        .context("`after` is not a valid mailbox cursor")?;
    if !id.is_empty() && !is_valid_mailbox_id(id) {
        anyhow::bail!("`after` is not a valid mailbox cursor");
    }
    Ok((created_at_ms, id.to_string()))
}

fn clamp_limit(limit: Option<usize>) -> usize {
    limit.unwrap_or(DEFAULT_PAGE_LIMIT).clamp(1, MAX_PAGE_LIMIT)
}

/// A bounded page of live envelopes addressed to `environment_id` that the same
/// environment has not marked consumed yet, ordered by `(created_at_ms, id)`
/// so the opaque cursor is a stable resume point.
pub(super) fn pending_envelopes(
    root: &Path,
    environment_id: &str,
    after: Option<&str>,
    limit: Option<usize>,
) -> anyhow::Result<Value> {
    if !is_valid_mailbox_id(environment_id) {
        anyhow::bail!("`environment_id` must be 1..={MAX_ID_BYTES} bytes of [A-Za-z0-9_-]");
    }
    let (after_created, after_id) = match after {
        Some(cursor) if !cursor.trim().is_empty() => decode_cursor(cursor.trim())?,
        _ => (-1, String::new()),
    };
    let limit = clamp_limit(limit);
    let now = now_ms();
    let conn = open_mailbox_store(root)?;
    let mut statement = conn.prepare(&format!(
        r#"
        SELECT id,
               CAST(COALESCE(json_extract(data, '$.created_at_ms'), 0) AS INTEGER) AS created_at_ms,
               data
        FROM "{MAILBOX_TABLE}"
        WHERE deleted = 0
          AND json_extract(data, '$.target_environment_id') = ?1
          AND (CAST(COALESCE(json_extract(data, '$.expires_at_ms'), 0) AS INTEGER) <= 0
               OR CAST(COALESCE(json_extract(data, '$.expires_at_ms'), 0) AS INTEGER) > ?2)
          AND (created_at_ms > ?3 OR (created_at_ms = ?3 AND id > ?4))
          AND NOT EXISTS (
                SELECT 1
                FROM json_each(COALESCE(json_extract(data, '$.consumed_by'), '[]'))
                WHERE json_each.value = ?1
              )
        ORDER BY created_at_ms ASC, id ASC
        LIMIT ?5
        "#
    ))?;
    let rows = statement.query_map(
        params![
            environment_id,
            now,
            after_created,
            after_id,
            (limit + 1) as i64
        ],
        |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, String>(2)?,
            ))
        },
    )?;

    let mut envelopes = Vec::new();
    let mut cursor_source = Vec::new();
    let mut has_more = false;
    for row in rows {
        let (id, created_at_ms, raw) = row?;
        if envelopes.len() == limit {
            has_more = true;
            break;
        }
        match serde_json::from_str::<Value>(&raw) {
            Ok(document) => {
                cursor_source.push((created_at_ms, id));
                envelopes.push(document);
            }
            Err(error) => {
                // log-and-continue: one unreadable row must never make the
                // whole mailbox page unreadable for the consumer.
                eprintln!("[workjet-mailbox] skipping unreadable envelope {id}: {error:#}");
            }
        }
    }
    let next_cursor = cursor_source
        .last()
        .map(|(created_at_ms, id)| encode_cursor(*created_at_ms, id));

    Ok(json!({
        "ok": true,
        "environment_id": environment_id,
        "envelopes": envelopes,
        "count": envelopes.len(),
        "has_more": has_more,
        "next_cursor": next_cursor,
    }))
}

/// Marks envelopes consumed by one environment. `consumed_by` is ADDITIVE: the
/// stored array and the request are unioned, never replaced, so two consumers
/// racing on the same envelope cannot erase each other's progress.
pub(super) fn mark_consumed(
    root: &Path,
    environment_id: &str,
    envelope_ids: &[String],
) -> anyhow::Result<Value> {
    if !is_valid_mailbox_id(environment_id) {
        anyhow::bail!("`environment_id` must be 1..={MAX_ID_BYTES} bytes of [A-Za-z0-9_-]");
    }
    if envelope_ids.is_empty() {
        anyhow::bail!("`envelope_ids` must not be empty");
    }
    if envelope_ids.len() > MAX_PAGE_LIMIT {
        anyhow::bail!("`envelope_ids` must not exceed {MAX_PAGE_LIMIT} entries per call");
    }
    for id in envelope_ids {
        if !is_valid_mailbox_id(id) {
            anyhow::bail!("`envelope_ids` entry `{id}` is not a valid envelope id");
        }
    }
    let now = now_ms();
    let conn = open_mailbox_store(root)?;
    let mut updated = Vec::new();
    let mut unchanged = Vec::new();
    let mut missing = Vec::new();
    for id in envelope_ids {
        let Some((deleted, mut document)) = load_document(&conn, id)? else {
            missing.push(id.clone());
            continue;
        };
        if deleted {
            missing.push(id.clone());
            continue;
        }
        let mut consumed: Vec<String> = document
            .get("consumed_by")
            .and_then(Value::as_array)
            .map(|entries| {
                entries
                    .iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if consumed.iter().any(|entry| entry == environment_id) {
            unchanged.push(id.clone());
            continue;
        }
        if consumed.len() >= MAX_CONSUMED_BY {
            eprintln!(
                "[workjet-mailbox] envelope {id} already lists {MAX_CONSUMED_BY} consumers; refusing to grow it"
            );
            unchanged.push(id.clone());
            continue;
        }
        consumed.push(environment_id.to_string());
        let mut seen = HashSet::new();
        consumed.retain(|entry| seen.insert(entry.clone()));
        document["consumed_by"] = Value::Array(consumed.into_iter().map(Value::String).collect());
        document["updated_at_ms"] = Value::from(now);
        write_document(&conn, id, &document, false, now)?;
        updated.push(id.clone());
    }
    Ok(json!({
        "ok": true,
        "environment_id": environment_id,
        "updated": updated,
        "unchanged": unchanged,
        "missing": missing,
    }))
}

// ---------------------------------------------------------------------------
// Expiry hygiene
// ---------------------------------------------------------------------------

/// Tombstones one bounded batch of envelopes past `expires_at_ms`.
///
/// Tombstoning (`deleted = 1` plus `_deleted` in the payload) rather than a raw
/// `DELETE` is what RxDB replication understands: a hard row delete would keep
/// the envelope alive on every peer that has not seen the removal. Returns the
/// number of envelopes tombstoned.
/// Builds the retirement document for one expired envelope.
///
/// # Why a tombstone keeps the envelope's required fields
///
/// A tombstone is a REPLICATED WRITE like any other, and the receiving peer runs
/// `rx_schema::validate_write_document` over it before persisting. That guard
/// rejects any document missing a `required` field with a 422 and DROPS it from
/// the batch — silently, as far as the sender is concerned. A bare
/// `{id, _deleted}` tombstone therefore retires the envelope locally and is
/// thrown away by every peer: the envelope stays live on every other machine
/// forever, and the row that proves it was retired never arrives. So the
/// tombstone carries the schema's required identity (`id`,
/// `target_environment_id`, `envelope_json`) plus the routing ids.
///
/// It deliberately does NOT carry `payload_json`. Dropping the opaque blob is
/// the whole point of retiring an envelope; the identity is small and bounded,
/// the payload is up to 200 KB.
///
/// A row whose stored JSON cannot be parsed still gets a minimal tombstone — a
/// corrupt row must still be retirable — and that one may indeed be refused by
/// a strict peer, which is strictly better than leaving it live locally.
fn tombstone_document(id: &str, stored_json: &str, now: i64) -> Value {
    let mut tombstone = json!({
        "id": id,
        "is_deleted": true,
        "_deleted": true,
        "updated_at_ms": now,
    });
    let Ok(previous) = serde_json::from_str::<Value>(stored_json) else {
        return tombstone;
    };
    for field in [
        "target_environment_id",
        "envelope_json",
        "source_workspace_id",
        "target_workspace_id",
        "source_environment_id",
        "created_at_ms",
        "expires_at_ms",
    ] {
        if let Some(value) = previous.get(field) {
            if !value.is_null() {
                tombstone[field] = value.clone();
            }
        }
    }
    tombstone
}

pub(super) fn sweep_expired_envelopes(
    root: &Path,
    now: i64,
    batch_limit: usize,
) -> anyhow::Result<usize> {
    let conn = open_mailbox_store(root)?;
    let mut statement = conn.prepare(&format!(
        r#"
        SELECT id, data
        FROM "{MAILBOX_TABLE}"
        WHERE deleted = 0
          AND CAST(COALESCE(json_extract(data, '$.expires_at_ms'), 0) AS INTEGER) > 0
          AND CAST(COALESCE(json_extract(data, '$.expires_at_ms'), 0) AS INTEGER) <= ?1
        ORDER BY id ASC
        LIMIT ?2
        "#
    ))?;
    let expired = statement
        .query_map(params![now, batch_limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?
        .collect::<Result<Vec<(String, String)>, _>>()?;
    drop(statement);

    let mut swept = 0usize;
    for (id, raw) in expired {
        let tombstone = tombstone_document(&id, &raw, now);
        match write_document(&conn, &id, &tombstone, true, now) {
            Ok(()) => swept += 1,
            // log-and-continue: one bad row must not abort the whole sweep.
            Err(error) => {
                eprintln!("[workjet-mailbox] failed to tombstone expired envelope {id}: {error:#}")
            }
        }
    }
    Ok(swept)
}

/// Starts the periodic expiry sweep.
///
/// Hooked into the daemon's existing background-cadence entry point
/// (`business_os::start_background_sync`, called once from
/// `service::…` at daemon start). No existing periodic loop could be reused
/// without editing an over-budget module, so this reuses the STARTUP point
/// instead of adding a second one.
pub(crate) fn start_expired_envelope_sweep(root: &Path) {
    let root = root.to_path_buf();
    std::thread::spawn(move || loop {
        match sweep_expired_envelopes(&root, now_ms(), SWEEP_BATCH_LIMIT) {
            Ok(swept) if swept > 0 => {
                eprintln!("[workjet-mailbox] tombstoned {swept} expired envelope(s)");
            }
            Ok(_) => {}
            Err(error) => {
                eprintln!("[workjet-mailbox] expired-envelope sweep failed: {error:#}");
            }
        }
        std::thread::sleep(Duration::from_secs(SWEEP_INTERVAL_SECS));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root() -> tempfile::TempDir {
        tempfile::tempdir().expect("temp root")
    }

    fn publish_request(id: &str, target: &str) -> Value {
        json!({
            "id": id,
            "source_workspace_id": "ws-source",
            "target_workspace_id": "ws-target",
            "source_environment_id": "env-source",
            "target_environment_id": target,
            "envelope_json": "{\"sig\":\"opaque\"}",
            "payload_json": "{\"body\":\"opaque\"}",
        })
    }

    #[test]
    fn mailbox_schema_parses_into_the_rxdb_schema_type() {
        let schema = mailbox_rx_schema();
        assert_eq!(schema.primary_key.primary_field(), "id");
        assert_eq!(i64::from(schema.version), MAILBOX_SCHEMA_VERSION);
        assert!(schema.indexes.is_empty());
        assert!(schema.additional_properties);
    }

    #[test]
    fn mailbox_collection_joins_the_registered_creator_map() {
        let creators = with_mailbox_collection(HashMap::new());
        assert!(creators.contains_key(MAILBOX_COLLECTION));
    }

    #[test]
    fn existing_creator_for_the_same_name_stays_authoritative() {
        let mut creators = HashMap::new();
        creators.insert(
            MAILBOX_COLLECTION.to_string(),
            RxCollectionCreator {
                schema: mailbox_rx_schema(),
                conflict_handler: None,
                options: HashMap::from([("marker".to_string(), json!(true))]),
            },
        );
        let creators = with_mailbox_collection(creators);
        assert!(creators[MAILBOX_COLLECTION].options.contains_key("marker"));
    }

    #[test]
    fn envelope_ids_are_bounded_and_charset_restricted() {
        assert!(is_valid_mailbox_id("env-1_A"));
        assert!(!is_valid_mailbox_id(""));
        assert!(!is_valid_mailbox_id("has space"));
        assert!(!is_valid_mailbox_id("has/slash"));
        assert!(!is_valid_mailbox_id("hät"));
        assert!(is_valid_mailbox_id(&"a".repeat(MAX_ID_BYTES)));
        assert!(!is_valid_mailbox_id(&"a".repeat(MAX_ID_BYTES + 1)));
    }

    #[test]
    fn document_validation_rejects_bad_bounds_and_accepts_a_valid_envelope() {
        let now = 1_700_000_000_000;
        let (id, document) = build_envelope_document(&publish_request("env-1", "target-a"), now)
            .expect("valid envelope");
        assert_eq!(id, "env-1");
        assert_eq!(document["target_environment_id"].as_str(), Some("target-a"));
        assert_eq!(document["created_at_ms"].as_i64(), Some(now));
        assert_eq!(
            document["expires_at_ms"].as_i64(),
            Some(now + DEFAULT_TTL_MS)
        );
        assert_eq!(document["consumed_by"].as_array().map(Vec::len), Some(0));

        let mut bad_id = publish_request("bad id", "target-a");
        assert!(build_envelope_document(&bad_id, now).is_err());
        bad_id["id"] = json!("env-1");
        bad_id["target_environment_id"] = json!("");
        assert!(build_envelope_document(&bad_id, now).is_err());

        let mut oversized_envelope = publish_request("env-2", "target-a");
        oversized_envelope["envelope_json"] = json!("x".repeat(MAX_ENVELOPE_JSON_BYTES + 1));
        assert!(build_envelope_document(&oversized_envelope, now).is_err());

        let mut oversized_payload = publish_request("env-3", "target-a");
        oversized_payload["payload_json"] = json!("x".repeat(MAX_PAYLOAD_JSON_BYTES + 1));
        assert!(build_envelope_document(&oversized_payload, now).is_err());

        let mut at_ceiling = publish_request("env-4", "target-a");
        at_ceiling["payload_json"] = json!("x".repeat(MAX_PAYLOAD_JSON_BYTES));
        let (_, document) =
            build_envelope_document(&at_ceiling, now).expect("ceiling payload fits");
        assert!(serde_json::to_vec(&document).expect("encode").len() <= MAX_DOCUMENT_BYTES);

        let mut bad_routing = publish_request("env-5", "target-a");
        bad_routing["source_workspace_id"] = json!("not valid");
        assert!(build_envelope_document(&bad_routing, now).is_err());

        let mut negative_time = publish_request("env-6", "target-a");
        negative_time["expires_at_ms"] = json!(-1);
        assert!(build_envelope_document(&negative_time, now).is_err());
    }

    #[test]
    fn publish_pending_consumed_round_trip_with_duplicates_and_paging() -> anyhow::Result<()> {
        let root = temp_root();
        let root = root.path();

        // Deterministic, strictly increasing creation stamps so the cursor
        // order is unambiguous. They must stay near `now`, because the default
        // TTL is relative to `created_at_ms` and `pending` hides expired rows.
        let base = now_ms() - 10_000;
        for index in 0..5 {
            let mut request = publish_request(&format!("env-{index}"), "target-a");
            request["created_at_ms"] = json!(base + index as i64);
            let result = publish_envelope(root, &request)?;
            assert_eq!(result["duplicate"].as_bool(), Some(false));
        }
        // A different target must not appear in target-a's page.
        publish_envelope(root, &publish_request("env-other", "target-b"))?;

        let duplicate = publish_envelope(root, &publish_request("env-0", "target-a"))?;
        assert_eq!(duplicate["duplicate"].as_bool(), Some(true));

        let page = pending_envelopes(root, "target-a", None, Some(2))?;
        assert_eq!(page["count"].as_u64(), Some(2));
        assert_eq!(page["has_more"].as_bool(), Some(true));
        assert_eq!(page["envelopes"][0]["id"].as_str(), Some("env-0"));
        assert_eq!(page["envelopes"][1]["id"].as_str(), Some("env-1"));
        let cursor = page["next_cursor"].as_str().expect("cursor").to_string();

        let page_two = pending_envelopes(root, "target-a", Some(&cursor), Some(2))?;
        assert_eq!(page_two["envelopes"][0]["id"].as_str(), Some("env-2"));
        assert_eq!(page_two["envelopes"][1]["id"].as_str(), Some("env-3"));

        let cursor_two = page_two["next_cursor"]
            .as_str()
            .expect("cursor")
            .to_string();
        let page_three = pending_envelopes(root, "target-a", Some(&cursor_two), Some(2))?;
        assert_eq!(page_three["count"].as_u64(), Some(1));
        assert_eq!(page_three["has_more"].as_bool(), Some(false));
        assert_eq!(page_three["envelopes"][0]["id"].as_str(), Some("env-4"));

        // consumed is additive and removes the envelope from that environment's
        // pending page while leaving it pending for a different environment.
        let consumed = mark_consumed(root, "target-a", &["env-0".to_string()])?;
        assert_eq!(consumed["updated"], json!(["env-0"]));
        let repeat = mark_consumed(root, "target-a", &["env-0".to_string()])?;
        assert_eq!(repeat["unchanged"], json!(["env-0"]));
        assert_eq!(repeat["updated"], json!([]));

        let after_consume = pending_envelopes(root, "target-a", None, Some(10))?;
        assert_eq!(after_consume["count"].as_u64(), Some(4));
        assert_eq!(after_consume["envelopes"][0]["id"].as_str(), Some("env-1"));

        let (_, stored) = load_document(&open_mailbox_store(root)?, "env-0")?.expect("stored");
        assert_eq!(stored["consumed_by"], json!(["target-a"]));
        let second = mark_consumed(root, "other-env", &["env-0".to_string()])?;
        assert_eq!(second["updated"], json!(["env-0"]));
        let (_, stored) = load_document(&open_mailbox_store(root)?, "env-0")?.expect("stored");
        assert_eq!(stored["consumed_by"], json!(["target-a", "other-env"]));

        let missing = mark_consumed(root, "target-a", &["env-missing".to_string()])?;
        assert_eq!(missing["missing"], json!(["env-missing"]));

        assert!(mark_consumed(root, "target-a", &[]).is_err());
        assert!(mark_consumed(root, "bad env", &["env-0".to_string()]).is_err());
        assert!(mark_consumed(root, "target-a", &["bad id".to_string()]).is_err());
        assert!(pending_envelopes(root, "bad env", None, None).is_err());
        assert!(pending_envelopes(root, "target-a", Some("!!!not-base64"), None).is_err());
        Ok(())
    }

    #[test]
    fn expiry_sweep_tombstones_only_expired_envelopes_in_bounded_batches() -> anyhow::Result<()> {
        let root = temp_root();
        let root = root.path();
        let now = now_ms();

        for index in 0..3 {
            let mut request = publish_request(&format!("old-{index}"), "target-a");
            request["created_at_ms"] = json!(now - 10_000);
            request["expires_at_ms"] = json!(now - 5_000);
            publish_envelope(root, &request)?;
        }
        let mut live = publish_request("live-1", "target-a");
        live["expires_at_ms"] = json!(now + 600_000);
        publish_envelope(root, &live)?;

        // Expired envelopes are invisible to `pending` even before the sweep.
        let page = pending_envelopes(root, "target-a", None, Some(10))?;
        assert_eq!(page["count"].as_u64(), Some(1));
        assert_eq!(page["envelopes"][0]["id"].as_str(), Some("live-1"));

        let swept = sweep_expired_envelopes(root, now, 2)?;
        assert_eq!(swept, 2, "the batch limit must bound one sweep pass");
        let swept = sweep_expired_envelopes(root, now, SWEEP_BATCH_LIMIT)?;
        assert_eq!(swept, 1);
        assert_eq!(sweep_expired_envelopes(root, now, SWEEP_BATCH_LIMIT)?, 0);

        let conn = open_mailbox_store(root)?;
        let (deleted, document) = load_document(&conn, "old-0")?.expect("tombstone row");
        assert!(deleted);
        assert_eq!(document["_deleted"].as_bool(), Some(true));
        let (deleted, _) = load_document(&conn, "live-1")?.expect("live row");
        assert!(!deleted);

        // A tombstoned envelope is neither pending nor consumable.
        assert_eq!(
            pending_envelopes(root, "target-a", None, Some(10))?["count"].as_u64(),
            Some(1)
        );
        let consumed = mark_consumed(root, "target-a", &["old-0".to_string()])?;
        assert_eq!(consumed["missing"], json!(["old-0"]));
        Ok(())
    }

    /// REGRESSION: every write must produce an RxDB-parseable `_rev` whose
    /// height grows. The original writer stamped `rev_<uuid>` and kept it out
    /// of the document, so a replicated peer accepted the first insert and then
    /// silently discarded every update — including the expiry tombstone, which
    /// left retired envelopes alive on the other machine forever.
    #[test]
    fn every_write_bumps_a_parseable_revision_height_inside_the_document() -> anyhow::Result<()> {
        let root = temp_root();
        let root = root.path();
        let conn = open_mailbox_store(root)?;

        publish_envelope(root, &publish_request("env-rev", "target-a"))?;
        let (_, inserted) = load_document(&conn, "env-rev")?.expect("inserted");
        let first = inserted["_rev"].as_str().expect("insert carries _rev");
        assert_eq!(first.split('-').next(), Some("1"));
        assert_eq!(
            first.matches('-').count(),
            1,
            "exactly one height separator"
        );
        assert_eq!(inserted["_deleted"].as_bool(), Some(false));
        assert!(inserted["_meta"]["lwt"].as_i64().is_some());

        mark_consumed(root, "target-a", &["env-rev".to_string()])?;
        let (_, updated) = load_document(&conn, "env-rev")?.expect("updated");
        assert_eq!(
            updated["_rev"]
                .as_str()
                .and_then(|rev| rev.split('-').next()),
            Some("2"),
            "an update must outrank the insert it replaces"
        );

        sweep_expired_envelopes(root, now_ms() + DEFAULT_TTL_MS + 1, 10)?;
        let (deleted, tombstone) = load_document(&conn, "env-rev")?.expect("tombstone");
        assert!(deleted);
        assert_eq!(
            tombstone["_rev"]
                .as_str()
                .and_then(|rev| rev.split('-').next()),
            Some("3"),
            "the tombstone must outrank the document it retires, or peers keep it alive"
        );
        assert_eq!(tombstone["_deleted"].as_bool(), Some(true));
        Ok(())
    }

    /// REGRESSION: a tombstone must survive the write-path validation every
    /// receiving peer runs. The original bare `{id, _deleted}` tombstone was
    /// rejected with a 422 and dropped from the replication batch, so an
    /// envelope retired on one machine stayed live on every other one — and the
    /// sender saw nothing wrong. Asserted against the REAL validator, not a
    /// restatement of the schema, so the two cannot drift apart.
    #[test]
    fn a_tombstone_passes_the_peer_write_validator_and_sheds_the_payload() -> anyhow::Result<()> {
        let root = temp_root();
        let root = root.path();
        let mut request = publish_request("env-tomb", "target-a");
        request["payload_json"] = json!("x".repeat(4096));
        request["source_environment_id"] = json!("source-env");
        publish_envelope(root, &request)?;

        let swept = sweep_expired_envelopes(root, now_ms() + DEFAULT_TTL_MS + 1, 10)?;
        assert_eq!(swept, 1);

        let (deleted, tombstone) =
            load_document(&open_mailbox_store(root)?, "env-tomb")?.expect("tombstone row");
        assert!(deleted);
        assert_eq!(tombstone["_deleted"].as_bool(), Some(true));
        assert_eq!(
            tombstone["target_environment_id"].as_str(),
            Some("target-a")
        );
        assert_eq!(
            tombstone["source_environment_id"].as_str(),
            Some("source-env")
        );
        assert!(
            tombstone.get("payload_json").is_none(),
            "a retired envelope must not keep carrying its payload"
        );

        let schema = mailbox_rx_schema();
        rxdb::rx_schema::validate_write_document(&schema, "id", &tombstone)
            .expect("a tombstone every peer drops is not a tombstone");
        Ok(())
    }

    #[test]
    fn cursors_round_trip_and_reject_tampering() {
        let cursor = encode_cursor(1_234, "env-9");
        assert_ne!(cursor, "1234:env-9", "the cursor must be opaque");
        assert_eq!(
            decode_cursor(&cursor).expect("decode"),
            (1_234, "env-9".to_string())
        );
        assert!(decode_cursor("not-a-cursor--").is_err());
        assert!(decode_cursor(&URL_SAFE_NO_PAD.encode("nope")).is_err());
        assert!(decode_cursor(&URL_SAFE_NO_PAD.encode("1:bad id")).is_err());
    }
}

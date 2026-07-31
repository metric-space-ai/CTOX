// Origin: CTOX
// License: AGPL-3.0-only
//
// Meeting bot adapter — joins video meetings (Google Meet, Microsoft Teams, Zoom)
// as a silent participant via Playwright, captures audio for transcription, monitors
// the meeting chat, and responds when @CTOX is mentioned.

use anyhow::{bail, Context, Result};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::ffi::OsStr;
use std::fs;
use std::io::{BufRead, BufReader, Write as IoWrite};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Stdio};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration as StdDuration, Instant, SystemTime, UNIX_EPOCH};

use rusqlite::OptionalExtension;

use crate::communication::adapters::{AdapterSyncCommandRequest, MeetingSendCommandRequest};
use crate::communication::runtime as communication_runtime;
use crate::inference::{engine, native_stt, runtime_env, supervisor};
use crate::mission::channels::{
    ensure_routing_rows_for_inbound, open_channel_db, refresh_thread, upsert_communication_message,
    UpsertMessage,
};

const DEFAULT_MEETING_STT_MODEL: &str = "engineai/Voxtral-Mini-4B-Realtime-2602";
const MEETING_XVFB_SERVER_ARGS: &str = "-screen 0 1920x1080x24 -ac +extension RANDR";

static MEETING_SYNC_FILE_CACHE: OnceLock<Mutex<BTreeMap<PathBuf, MeetingSyncFileState>>> =
    OnceLock::new();

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MeetingSessionFileStamp {
    len: u64,
    modified_ns: u128,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MeetingSyncFileState {
    stamp: MeetingSessionFileStamp,
    active: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeetingSessionStatus {
    Joining,
    Running,
    Active,
    JoinFailed,
    Ended,
}

impl MeetingSessionStatus {
    fn parse(value: &str) -> Result<Self> {
        match value {
            "joining" => Ok(Self::Joining),
            "running" => Ok(Self::Running),
            "active" => Ok(Self::Active),
            "join_failed" => Ok(Self::JoinFailed),
            "ended" => Ok(Self::Ended),
            other => bail!("unknown meeting session status `{other}`"),
        }
    }

    fn from_session_value(session: &Value) -> Result<Self> {
        let status = session
            .get("status")
            .and_then(Value::as_str)
            .context("meeting session status must be a string")?;
        Self::parse(status)
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Joining => "joining",
            Self::Running => "running",
            Self::Active => "active",
            Self::JoinFailed => "join_failed",
            Self::Ended => "ended",
        }
    }

    fn is_running(self) -> bool {
        matches!(self, Self::Joining | Self::Running | Self::Active)
    }
}

// ---------------------------------------------------------------------------
// Public adapter interface (sync / send / service_sync)
// ---------------------------------------------------------------------------

/// Sync active meeting sessions — ingest new chat messages into the SQLite
/// communication_messages table, exactly like email/jami sync does.
/// Each chat message becomes a row with channel="meeting",
/// thread_key=session_id, direction="inbound".
pub(crate) fn sync(
    root: &Path,
    _runtime: &BTreeMap<String, String>,
    request: &AdapterSyncCommandRequest<'_>,
) -> Result<Value> {
    let session_dirs = existing_meeting_session_dirs(root);
    if session_dirs.is_empty() {
        return Ok(json!({"ok": true, "active_sessions": 0, "ingested": 0}));
    }
    let db_path = request.db_path;
    let mut conn = open_channel_db(db_path)?;
    let mut active = 0u64;
    let mut ingested = 0u64;
    let mut skipped_unchanged = 0u64;
    let mut session_files_seen = 0u64;
    let mut seen_session_files = Vec::new();
    let account_key = "meeting:system";

    for sessions_dir in session_dirs {
        let Ok(entries) = fs::read_dir(&sessions_dir) else {
            continue;
        };
        for entry in entries {
            let Ok(entry) = entry else { continue };
            let path = entry.path();
            if path.extension().and_then(|ext| ext.to_str()) != Some("json") {
                continue;
            }
            let Some(stamp) = meeting_session_file_stamp(&path) else {
                continue;
            };
            session_files_seen += 1;
            seen_session_files.push(path.clone());
            if let Some(cached) = cached_meeting_session_file_state(&path, stamp) {
                if cached.active {
                    active += 1;
                }
                skipped_unchanged += 1;
                continue;
            }
            let Ok(contents) = fs::read_to_string(&path) else {
                continue;
            };
            let Ok(mut session) = serde_json::from_str::<Value>(&contents) else {
                continue;
            };
            let status = MeetingSessionStatus::from_session_value(&session)
                .with_context(|| format!("invalid meeting session status at {}", path.display()))?;
            if !status.is_running() {
                remember_meeting_session_file_state(&path, stamp, false);
                continue;
            }
            active += 1;
            let mut session_ingested = 0u64;

            let session_id = session
                .get("session_id")
                .and_then(Value::as_str)
                .unwrap_or("")
                .to_string();
            let provider = session
                .get("provider")
                .and_then(Value::as_str)
                .unwrap_or("unknown")
                .to_string();

            // Read chat messages from the session JSON and ingest any not yet in SQLite
            let chat_messages = session
                .get("chat_messages")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();

            for msg in &chat_messages {
                let sender = msg
                    .get("sender")
                    .and_then(Value::as_str)
                    .unwrap_or("Unknown");
                let text = msg.get("text").and_then(Value::as_str).unwrap_or("");
                let timestamp = msg.get("timestamp").and_then(Value::as_str).unwrap_or("");
                if text.is_empty() {
                    continue;
                }
                if session_value_is_own_message(&session, sender, text) {
                    continue;
                }

                // Stable message_key prevents re-ingesting the same chat line
                let message_key = format!(
                    "meeting::{}::{}",
                    session_id,
                    stable_digest(&format!("{sender}:{text}:{timestamp}"))
                );
                let is_mention = MeetingSession::is_mention(text);
                let known_message = communication_message_exists(&conn, &message_key)?;
                if known_message {
                    if is_mention
                        && !session
                            .get("mention_ack_sent")
                            .and_then(Value::as_bool)
                            .unwrap_or(false)
                    {
                        let ack_text = first_mention_ack_text();
                        submit_meeting_outbound_message(
                            &mut conn,
                            &session,
                            &session_id,
                            &provider,
                            ack_text,
                            "ctox_first_mention_ack",
                        )?;
                        if let Some(object) = session.as_object_mut() {
                            object.insert("mention_ack_sent".to_string(), Value::Bool(true));
                            object.insert(
                                "mention_ack_sent_at".to_string(),
                                Value::String(now_iso_string()),
                            );
                        }
                        let _ = fs::write(&path, serde_json::to_string_pretty(&session)?);
                    }
                    continue;
                }

                let observed_at = if timestamp.is_empty() {
                    now_iso_string()
                } else {
                    timestamp.to_string()
                };

                if is_mention
                    && !session
                        .get("mention_ack_sent")
                        .and_then(Value::as_bool)
                        .unwrap_or(false)
                {
                    let ack_text = first_mention_ack_text();
                    submit_meeting_outbound_message(
                        &mut conn,
                        &session,
                        &session_id,
                        &provider,
                        ack_text,
                        "ctox_first_mention_ack",
                    )?;
                    if let Some(object) = session.as_object_mut() {
                        object.insert("mention_ack_sent".to_string(), Value::Bool(true));
                        object.insert(
                            "mention_ack_sent_at".to_string(),
                            Value::String(now_iso_string()),
                        );
                    }
                    let _ = fs::write(&path, serde_json::to_string_pretty(&session)?);
                }
                let transcript_snapshot = session_transcript_snapshot(&session, 12);
                let chat_snapshot = session_chat_snapshot(&session, 20);
                let body_text = if is_mention {
                    render_meeting_mention_inbound_body(
                        &session_id,
                        &provider,
                        sender,
                        text,
                        timestamp,
                        &transcript_snapshot,
                        &chat_snapshot,
                    )
                } else {
                    text.to_string()
                };
                let preview = clip_chars(&body_text, 120);
                let metadata = json!({
                    "provider": &provider,
                    "session_id": &session_id,
                    "source": "meeting_chat",
                    "is_mention": is_mention,
                    "skill": if is_mention { "meeting-participant" } else { "" },
                    "priority": if is_mention { "urgent" } else { "normal" },
                    "transcript_chunk_count": session
                        .get("transcript_chunk_count")
                        .and_then(Value::as_u64)
                        .unwrap_or_else(|| session
                            .get("transcript_chunks")
                            .and_then(Value::as_array)
                            .map(|items| items.len() as u64)
                            .unwrap_or(0)),
                    "chat_message_count": session
                        .get("chat_message_count")
                        .and_then(Value::as_u64)
                        .unwrap_or_else(|| chat_messages.len() as u64),
                    "transcript_snapshot": transcript_snapshot,
                    "chat_snapshot": chat_snapshot,
                });

                upsert_communication_message(
                    &mut conn,
                    UpsertMessage {
                        message_key: &message_key,
                        channel: "meeting",
                        account_key,
                        thread_key: &session_id,
                        remote_id: &message_key,
                        direction: "inbound",
                        folder_hint: "chat",
                        sender_display: sender,
                        sender_address: sender,
                        recipient_addresses_json: "[]",
                        cc_addresses_json: "[]",
                        bcc_addresses_json: "[]",
                        subject: &format!("{} meeting chat", provider),
                        preview: &preview,
                        body_text: &body_text,
                        body_html: "",
                        raw_payload_ref: "",
                        trust_level: "internal",
                        status: "received",
                        seen: false,
                        has_attachments: false,
                        external_created_at: &observed_at,
                        observed_at: &observed_at,
                        metadata_json: &serde_json::to_string(&metadata)?,
                    },
                )?;
                session_ingested += 1;
                ingested += 1;
            }

            if session_ingested > 0 {
                let _ = refresh_thread(&mut conn, &session_id);
            }
            let latest_stamp = meeting_session_file_stamp(&path).unwrap_or(stamp);
            remember_meeting_session_file_state(&path, latest_stamp, true);
        }
    }
    retain_seen_meeting_session_file_cache(&seen_session_files);

    if ingested > 0 {
        ensure_routing_rows_for_inbound(&conn)?;
    }

    Ok(json!({
        "ok": true,
        "active_sessions": active,
        "ingested": ingested,
        "session_files_seen": session_files_seen,
        "skipped_unchanged_sessions": skipped_unchanged,
    }))
}

/// Send a chat message to a running meeting session.
/// 1. Append the command to the runner's durable command file.
/// 2. Record it as `submitted`; only a runner `chat_sent` event may promote it to `sent`.
pub(crate) fn send(
    root: &Path,
    _runtime: &BTreeMap<String, String>,
    request: &MeetingSendCommandRequest<'_>,
) -> Result<Value> {
    let session_path = meeting_session_file(root, request.session_id);
    if !session_path.exists() {
        bail!(
            "meeting session {} not found at {}",
            request.session_id,
            session_path.display()
        );
    }
    let contents = fs::read_to_string(&session_path)?;
    let session: Value = serde_json::from_str(&contents)?;
    let status = MeetingSessionStatus::from_session_value(&session).with_context(|| {
        format!(
            "invalid meeting session status at {}",
            session_path.display()
        )
    })?;
    if !status.is_running() {
        bail!(
            "meeting session {} is not running (status={})",
            request.session_id,
            status.as_str()
        );
    }
    let provider = session
        .get("provider")
        .and_then(Value::as_str)
        .unwrap_or("unknown");
    let observed_at = now_iso_string();
    let message_key =
        meeting_outbound_message_key(request.session_id, "ctox_reply", request.body, &observed_at);

    write_chat_command_to_session(&session, request.body, Some(&message_key))?;

    let mut conn = open_channel_db(request.db_path)?;
    record_meeting_outbound_message(
        &mut conn,
        request.session_id,
        provider,
        request.body,
        "ctox_reply",
        &message_key,
        &observed_at,
    )?;

    Ok(json!({
        "ok": true,
        "status": "submitted",
        "session_id": request.session_id,
        "message_key": message_key,
        "delivery": {
            "confirmed": false,
            "state": "submitted_to_meeting_runner",
            "detail": "submitted means the command was appended to the runner command file; participant-visible delivery is not yet confirmed"
        }
    }))
}

fn first_mention_ack_text() -> &'static str {
    "Ich habe die Frage gesehen und antworte hier im Chat. Das kann einen Augenblick dauern, weil mir Echtzeit-Antworten leider noch nicht zuverlaessig moeglich sind."
}

fn meeting_outbound_message_key(
    session_id: &str,
    source: &str,
    body: &str,
    observed_at: &str,
) -> String {
    format!(
        "meeting::{session_id}::out::{}",
        stable_digest(&format!("{source}:{body}:{observed_at}"))
    )
}

fn write_chat_command_to_session(
    session: &Value,
    text: &str,
    message_key: Option<&str>,
) -> Result<()> {
    let stdin_path = session
        .get("stdin_pipe")
        .and_then(Value::as_str)
        .filter(|path| !path.trim().is_empty())
        .context("meeting session has no runner command file")?;
    let mut command = json!({"action": "send_chat", "text": text});
    if let Some(message_key) = message_key {
        command["message_key"] = Value::String(message_key.to_string());
    }
    let mut file = fs::OpenOptions::new()
        .append(true)
        .open(stdin_path)
        .with_context(|| format!("failed to open meeting runner command file {stdin_path}"))?;
    writeln!(file, "{command}")
        .with_context(|| format!("failed to write meeting runner command file {stdin_path}"))?;
    file.flush()
        .with_context(|| format!("failed to flush meeting runner command file {stdin_path}"))?;
    Ok(())
}

fn append_line(path: &Path, line: &str) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("failed to open {}", path.display()))?;
    writeln!(file, "{line}")?;
    Ok(())
}

fn recent_direct_speaker(signal: Option<&SpeakerSignal>) -> Option<&SpeakerSignal> {
    let signal = signal?;
    let speaker = signal.speaker_display.trim();
    if speaker.is_empty() || speaker.eq_ignore_ascii_case("unknown") {
        return None;
    }
    let ts = DateTime::parse_from_rfc3339(&signal.timestamp).ok()?;
    let age = Utc::now().signed_duration_since(ts.with_timezone(&Utc));
    if age <= Duration::seconds(45) {
        Some(signal)
    } else {
        None
    }
}

fn session_value_is_own_message(session: &Value, sender: &str, text: &str) -> bool {
    let bot_name = session
        .get("bot_name")
        .and_then(Value::as_str)
        .unwrap_or("INF Yoda Notetaker");
    let outbound = session
        .get("outbound_chat_texts")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    is_own_message_text(bot_name, &outbound, sender, text)
}

fn meeting_session_file_stamp(path: &Path) -> Option<MeetingSessionFileStamp> {
    let metadata = fs::metadata(path).ok()?;
    if !metadata.is_file() {
        return None;
    }
    let modified_ns = metadata
        .modified()
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map(|duration| duration.as_nanos())
        .unwrap_or(0);
    Some(MeetingSessionFileStamp {
        len: metadata.len(),
        modified_ns,
    })
}

fn cached_meeting_session_file_state(
    path: &Path,
    stamp: MeetingSessionFileStamp,
) -> Option<MeetingSyncFileState> {
    let cache = MEETING_SYNC_FILE_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    let cache = cache.lock().ok()?;
    let state = cache.get(path)?;
    (state.stamp == stamp).then_some(*state)
}

fn remember_meeting_session_file_state(path: &Path, stamp: MeetingSessionFileStamp, active: bool) {
    let cache = MEETING_SYNC_FILE_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(mut cache) = cache.lock() {
        cache.insert(path.to_path_buf(), MeetingSyncFileState { stamp, active });
    }
}

fn retain_seen_meeting_session_file_cache(seen_session_files: &[PathBuf]) {
    let cache = MEETING_SYNC_FILE_CACHE.get_or_init(|| Mutex::new(BTreeMap::new()));
    if let Ok(mut cache) = cache.lock() {
        cache.retain(|path, _| seen_session_files.iter().any(|seen| seen == path) || path.exists());
    }
}

fn communication_message_exists(conn: &rusqlite::Connection, message_key: &str) -> Result<bool> {
    Ok(conn
        .query_row(
            "SELECT 1 FROM communication_messages WHERE message_key = ?1 LIMIT 1",
            [message_key],
            |_| Ok(()),
        )
        .optional()?
        .is_some())
}

fn submit_meeting_outbound_message(
    conn: &mut rusqlite::Connection,
    session: &Value,
    session_id: &str,
    provider: &str,
    body: &str,
    source: &str,
) -> Result<String> {
    let observed_at = now_iso_string();
    let message_key = meeting_outbound_message_key(session_id, source, body, &observed_at);
    write_chat_command_to_session(session, body, Some(&message_key))?;
    record_meeting_outbound_message(
        conn,
        session_id,
        provider,
        body,
        source,
        &message_key,
        &observed_at,
    )?;
    Ok(message_key)
}

fn record_meeting_outbound_message(
    conn: &mut rusqlite::Connection,
    session_id: &str,
    provider: &str,
    body: &str,
    source: &str,
    message_key: &str,
    observed_at: &str,
) -> Result<()> {
    let metadata = json!({
        "provider": provider,
        "session_id": session_id,
        "source": source,
        "delivery": {
            "confirmed": false,
            "state": "submitted_to_meeting_runner",
            "detail": "submitted means the command was appended to the runner command file; participant-visible delivery is not yet confirmed",
        },
    });
    let preview = clip_chars(body, 120);
    upsert_communication_message(
        conn,
        UpsertMessage {
            message_key,
            channel: "meeting",
            account_key: "meeting:system",
            thread_key: session_id,
            remote_id: message_key,
            direction: "outbound",
            folder_hint: "submitted",
            sender_display: "INF Yoda Notetaker",
            sender_address: "ctox@local",
            recipient_addresses_json: "[]",
            cc_addresses_json: "[]",
            bcc_addresses_json: "[]",
            subject: &format!("{} meeting chat reply", provider),
            preview: &preview,
            body_text: body,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "internal",
            status: "submitted",
            seen: true,
            has_attachments: false,
            external_created_at: observed_at,
            observed_at,
            metadata_json: &serde_json::to_string(&metadata)?,
        },
    )?;
    refresh_thread(conn, session_id)?;
    Ok(())
}

fn confirm_meeting_outbound_message(
    root: &Path,
    session_id: &str,
    message_key: &str,
) -> Result<bool> {
    let db_path = root.join("runtime/ctox.sqlite3");
    for attempt in 0..20 {
        if db_path.exists() {
            let mut conn = open_channel_db(&db_path)?;
            let row = conn
                .query_row(
                    "SELECT status, metadata_json
                     FROM communication_messages
                     WHERE message_key = ?1
                       AND channel = 'meeting'
                       AND direction = 'outbound'",
                    [message_key],
                    |row| Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?)),
                )
                .optional()?;
            if let Some((status, metadata_json)) = row {
                if status == "sent" {
                    return Ok(true);
                }
                if status != "submitted" {
                    return Ok(false);
                }
                let confirmed_at = now_iso_string();
                let mut metadata =
                    serde_json::from_str::<Value>(&metadata_json).unwrap_or_else(|_| json!({}));
                if let Some(object) = metadata.as_object_mut() {
                    object.insert(
                        "delivery".to_string(),
                        json!({
                            "confirmed": true,
                            "state": "runner_confirmed_sent",
                            "confirmed_at": confirmed_at,
                        }),
                    );
                }
                let changed = conn.execute(
                    "UPDATE communication_messages
                     SET status = 'sent',
                         folder_hint = 'sent',
                         observed_at = ?2,
                         metadata_json = ?3
                     WHERE message_key = ?1
                       AND status = 'submitted'",
                    rusqlite::params![message_key, confirmed_at, serde_json::to_string(&metadata)?],
                )?;
                if changed > 0 {
                    refresh_thread(&mut conn, session_id)?;
                    return Ok(true);
                }
            }
        }
        if attempt < 19 {
            std::thread::sleep(StdDuration::from_millis(50));
        }
    }
    Ok(false)
}

fn render_meeting_mention_inbound_body(
    session_id: &str,
    provider: &str,
    sender: &str,
    text: &str,
    timestamp: &str,
    transcript_snapshot: &str,
    chat_snapshot: &str,
) -> String {
    let timestamp = if timestamp.trim().is_empty() {
        "(unknown)"
    } else {
        timestamp
    };
    let transcript = if transcript_snapshot.trim().is_empty() {
        "(Noch kein Live-Transcript verfuegbar. Falls STT offline ist, antworte nur auf Basis von Chat und explizit bekannten Kontext.)"
    } else {
        transcript_snapshot
    };
    let chat = if chat_snapshot.trim().is_empty() {
        "(keine vorherigen Chatnachrichten)"
    } else {
        chat_snapshot
    };
    format!(
        "@CTOX Meeting-Chat-Erwaehnung\n\
         Provider: {provider}\n\
         Session: {session_id}\n\
         Sender: {sender}\n\
         Timestamp: {timestamp}\n\
         Nachricht: {text}\n\n\
         Live-Transcript bisher (neueste Chunks):\n{transcript}\n\n\
         Meeting-Chat bisher:\n{chat}\n\n\
         Antworte kurz im Meeting-Chat. Wenn das Transcript fuer die Frage nicht ausreicht, sage das knapp und frage nach der fehlenden Information."
    )
}

fn session_transcript_snapshot(session: &Value, max_chunks: usize) -> String {
    if let Some(snapshot) = session_transcript_segment_snapshot(session, max_chunks) {
        return snapshot;
    }
    let chunks = session
        .get("transcript_chunks")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .map(str::trim)
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let start = chunks.len().saturating_sub(max_chunks);
    chunks[start..].join("\n")
}

fn session_transcript_segment_snapshot(session: &Value, max_segments: usize) -> Option<String> {
    let segments = session
        .get("transcript_segments")
        .and_then(Value::as_array)?
        .iter()
        .filter_map(render_transcript_segment_value)
        .collect::<Vec<_>>();
    if segments.is_empty() {
        return None;
    }
    let start = segments.len().saturating_sub(max_segments);
    Some(segments[start..].join("\n"))
}

fn render_transcript_segment_value(segment: &Value) -> Option<String> {
    let text = segment.get("text").and_then(Value::as_str)?.trim();
    if text.is_empty() {
        return None;
    }
    let speaker = segment
        .get("speaker_display")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("unknown");
    let timestamp = segment
        .get("timestamp")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("unknown");
    let source = segment
        .get("source")
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("stt");
    let confidence = segment
        .get("confidence")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    Some(format!(
        "[{timestamp}] {speaker}: {text} [source={source} confidence={confidence:.2}]"
    ))
}

fn session_chat_snapshot(session: &Value, max_messages: usize) -> String {
    let messages = session
        .get("chat_messages")
        .and_then(Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|message| {
                    let sender = message
                        .get("sender")
                        .and_then(Value::as_str)
                        .unwrap_or("Unknown");
                    let text = message.get("text").and_then(Value::as_str).unwrap_or("");
                    if text.trim().is_empty() {
                        return None;
                    }
                    let timestamp = message
                        .get("timestamp")
                        .and_then(Value::as_str)
                        .unwrap_or("");
                    Some(if timestamp.trim().is_empty() {
                        format!("{sender}: {text}")
                    } else {
                        format!("[{timestamp}] {sender}: {text}")
                    })
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    let start = messages.len().saturating_sub(max_messages);
    messages[start..].join("\n")
}

fn clip_chars(value: &str, max_chars: usize) -> String {
    value
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .chars()
        .take(max_chars)
        .collect()
}

struct MeetingSttRuntimeGuard {
    root: PathBuf,
    started_by_meeting: bool,
    restore_values: BTreeMap<String, Option<String>>,
    start_error: Option<String>,
    finished: bool,
}

impl MeetingSttRuntimeGuard {
    fn ensure_for_meeting(root: &Path) -> Self {
        let mut guard = Self {
            root: root.to_path_buf(),
            started_by_meeting: false,
            restore_values: BTreeMap::new(),
            start_error: None,
            finished: false,
        };
        if check_engine_reachable(root) {
            return guard;
        }
        if let Err(err) = guard.prepare_runtime_config() {
            guard.start_error = Some(err.to_string());
            return guard;
        }
        match supervisor::ensure_auxiliary_backend_ready(root, engine::AuxiliaryRole::Stt, false) {
            Ok(()) => {
                if check_engine_reachable(root) {
                    guard.started_by_meeting = true;
                    eprintln!("[meeting] STT runtime auto-started for this meeting");
                } else {
                    guard.start_error = Some(
                        "STT backend launch completed but the transcription transport is still unavailable"
                            .to_string(),
                    );
                }
            }
            Err(err) => {
                guard.start_error = Some(err.to_string());
            }
        }
        guard
    }

    fn prepare_runtime_config(&mut self) -> Result<()> {
        let mut env_map = runtime_env::load_runtime_env_map(&self.root).unwrap_or_default();
        self.set_runtime_key(
            &mut env_map,
            "CTOX_ENABLE_STT_BACKEND",
            Some("1".to_string()),
        );
        let current_model = env_map
            .get("CTOX_STT_MODEL")
            .map(String::as_str)
            .unwrap_or("");
        if normalize_meeting_stt_model(Some(current_model)) != current_model.trim() {
            self.set_runtime_key(
                &mut env_map,
                "CTOX_STT_MODEL",
                Some(DEFAULT_MEETING_STT_MODEL.to_string()),
            );
        }
        runtime_env::save_runtime_env_map(&self.root, &env_map)
    }

    fn set_runtime_key(
        &mut self,
        env_map: &mut BTreeMap<String, String>,
        key: &'static str,
        value: Option<String>,
    ) {
        self.restore_values
            .entry(key.to_string())
            .or_insert_with(|| env_map.get(key).cloned());
        match value {
            Some(value) => {
                env_map.insert(key.to_string(), value);
            }
            None => {
                env_map.remove(key);
            }
        }
    }

    fn finish(&mut self) {
        if self.finished {
            return;
        }
        self.finished = true;
        if self.started_by_meeting {
            if let Err(err) =
                supervisor::release_auxiliary_backend(&self.root, engine::AuxiliaryRole::Stt)
            {
                eprintln!("[meeting] warning: failed to stop meeting STT runtime: {err}");
            } else {
                eprintln!("[meeting] STT runtime stopped after meeting");
            }
        }
        if !self.restore_values.is_empty() {
            if let Err(err) = self.restore_runtime_config() {
                eprintln!("[meeting] warning: failed to restore STT runtime config: {err}");
            }
        }
    }

    fn restore_runtime_config(&self) -> Result<()> {
        let mut env_map = runtime_env::load_runtime_env_map(&self.root).unwrap_or_default();
        for (key, value) in &self.restore_values {
            match value {
                Some(value) => {
                    env_map.insert(key.clone(), value.clone());
                }
                None => {
                    env_map.remove(key);
                }
            }
        }
        runtime_env::save_runtime_env_map(&self.root, &env_map)
    }
}

impl Drop for MeetingSttRuntimeGuard {
    fn drop(&mut self) {
        self.finish();
    }
}

fn is_disabled_selector(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "" | "0" | "false" | "off" | "none" | "disabled"
    )
}

/// Service sync — delegates to sync() with proper db_path.
pub(crate) fn service_sync(
    root: &Path,
    settings: &BTreeMap<String, String>,
) -> Result<Option<Value>> {
    let db_path = root.join("runtime/ctox.sqlite3");
    let request = AdapterSyncCommandRequest {
        db_path: &db_path,
        passthrough_args: &[],
        skip_flags: &[],
    };
    Ok(Some(sync(root, settings, &request)?))
}

// ---------------------------------------------------------------------------
// CLI command handler
// ---------------------------------------------------------------------------

pub fn handle_meeting_command(root: &Path, args: &[String]) -> Result<()> {
    let command = args.first().map(String::as_str).unwrap_or("");
    match command {
        "join" => {
            let url = args
                .get(1)
                .context("usage: ctox meeting join <url> [--name <bot-name>]")?;
            let bot_name = find_flag_value(args, "--name").unwrap_or("INF Yoda Notetaker");
            let runtime = crate::communication::gateway::runtime_settings_from_root(
                root,
                crate::communication::gateway::CommunicationAdapterKind::Meeting,
            );
            let mut config = MeetingSessionConfig::from_runtime(root, url, &runtime)?;
            if bot_name != "INF Yoda Notetaker" {
                config.bot_name = bot_name.to_string();
            }
            let result = run_meeting_session(root, &config)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        "schedule" => {
            let url = args
                .get(1)
                .context("usage: ctox meeting schedule <url> --time <ISO-8601>")?;
            let time = find_flag_value(args, "--time").context("--time <ISO-8601> is required")?;
            let bot_name = find_flag_value(args, "--name").unwrap_or("INF Yoda Notetaker");
            let result = schedule_meeting_join(root, url, time, bot_name)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        "cancel" => {
            let url = args.get(1).context("usage: ctox meeting cancel <url>")?;
            let result = cancel_meeting_join(root, url)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        "dump-script" => {
            let url = args
                .get(1)
                .context("usage: ctox meeting dump-script <url>")?;
            let runtime = crate::communication::gateway::runtime_settings_from_root(
                root,
                crate::communication::gateway::CommunicationAdapterKind::Meeting,
            );
            let config = MeetingSessionConfig::from_runtime(root, url, &runtime)?;
            let script = build_meeting_runner_script(&config)?;
            print!("{script}");
            Ok(())
        }
        "status" => {
            let mut sessions = Vec::new();
            for sessions_dir in existing_meeting_session_dirs(root) {
                for entry in fs::read_dir(&sessions_dir)? {
                    let entry = entry?;
                    if entry.path().extension().and_then(|e| e.to_str()) == Some("json") {
                        let path = entry.path();
                        let contents = fs::read_to_string(&path)
                            .with_context(|| format!("read meeting session {}", path.display()))?;
                        let session =
                            serde_json::from_str::<Value>(&contents).with_context(|| {
                                format!("parse meeting session JSON at {}", path.display())
                            })?;
                        MeetingSessionStatus::from_session_value(&session).with_context(|| {
                            format!("invalid meeting session status at {}", path.display())
                        })?;
                        sessions.push(session);
                    }
                }
            }
            println!(
                "{}",
                serde_json::to_string_pretty(&json!({
                    "ok": true,
                    "sessions": sessions,
                }))?
            );
            Ok(())
        }
        "transcript" => {
            let session_id = args
                .get(1)
                .context("usage: ctox meeting transcript <session_id>")?;
            let result = load_meeting_transcript(root, session_id)?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        "simulate" => {
            let result = simulate_meeting_session(root, &args[1..])?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            Ok(())
        }
        "preflight-realtime" => {
            let result = preflight_realtime_meeting(root, &args[1..])?;
            println!("{}", serde_json::to_string_pretty(&result)?);
            if !result.get("ok").and_then(Value::as_bool).unwrap_or(false) {
                bail!("meeting realtime preflight failed; refusing live Teams test");
            }
            Ok(())
        }
        _ => {
            println!(
                "usage: ctox meeting <join|schedule|cancel|status|transcript|simulate|preflight-realtime> [args]"
            );
            println!();
            println!("  join <url> [--name <bot-name>]       Join a meeting now");
            println!("  schedule <url> --time <ISO-8601>     Schedule a future join");
            println!("  cancel <url>                         Cancel a scheduled join");
            println!("  status                               Show active/scheduled sessions");
            println!("  transcript <session_id>              Print transcript + chatlog as JSON");
            println!(
                "  simulate [--audio <wav>]... [--transcript <text>]... [--chat <sender:text>]..."
            );
            println!(
                "  preflight-realtime [--audio <wav>] [--foreign-audio <wav>]   Verify isolated Teams audio + Mistral realtime before live use"
            );
            Ok(())
        }
    }
}

fn find_flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn find_flag_values<'a>(args: &'a [String], flag: &str) -> Vec<&'a str> {
    args.iter()
        .enumerate()
        .filter_map(|(idx, value)| {
            if value == flag {
                args.get(idx + 1).map(String::as_str)
            } else {
                None
            }
        })
        .collect()
}

fn simulate_meeting_session(root: &Path, args: &[String]) -> Result<Value> {
    let provider = match find_flag_value(args, "--provider")
        .unwrap_or("google")
        .trim()
        .to_ascii_lowercase()
        .as_str()
    {
        "google" | "meet" | "google-meet" => MeetingProvider::GoogleMeet,
        "microsoft" | "teams" | "microsoft-teams" => MeetingProvider::MicrosoftTeams,
        "zoom" => MeetingProvider::Zoom,
        other => bail!("unsupported --provider `{other}`; expected google, teams, or zoom"),
    };
    let meeting_url = find_flag_value(args, "--url")
        .map(ToOwned::to_owned)
        .unwrap_or_else(|| match provider {
            MeetingProvider::GoogleMeet => "https://meet.google.com/demo-meet-test".to_string(),
            MeetingProvider::MicrosoftTeams => "https://teams.microsoft.com/meet/demo".to_string(),
            MeetingProvider::Zoom => "https://zoom.us/j/123456789".to_string(),
        });
    let bot_name = find_flag_value(args, "--name").unwrap_or("INF Yoda Notetaker");
    let config = MeetingSessionConfig {
        root: root.to_path_buf(),
        meeting_url,
        provider,
        bot_name: bot_name.to_string(),
        max_duration_minutes: 60,
        audio_chunk_seconds: 3,
        stt_model: String::new(),
        realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
        mistral_api_key: None,
    };
    let mut session = MeetingSession::new(&config);
    session.status = MeetingSessionStatus::Ended;
    session.ended_at = Some(now_iso_string());

    for transcript in find_flag_values(args, "--transcript") {
        let transcript = transcript.trim();
        if !transcript.is_empty() {
            session.push_stt_transcript(transcript.to_string(), None);
        }
    }
    for chat in find_flag_values(args, "--chat") {
        let (sender, text) = chat
            .split_once(':')
            .map(|(sender, text)| (sender.trim(), text.trim()))
            .unwrap_or(("Participant", chat.trim()));
        if !text.is_empty() {
            session.chat_messages.push(ChatMessage {
                sender: if sender.is_empty() {
                    "Participant".to_string()
                } else {
                    sender.to_string()
                },
                text: text.to_string(),
                timestamp: now_iso_string(),
            });
        }
    }
    for audio_path in find_flag_values(args, "--audio") {
        match persist_audio_chunk(root, &session.session_id, audio_path) {
            Some(path) => session.pending_audio_chunks.push(path),
            None => eprintln!("[meeting] warning: could not persist fixture audio {audio_path}"),
        }
    }

    session.save(root)?;
    let finalization = finalize_meeting(root, &session, &config)?;
    Ok(json!({
        "ok": true,
        "session_id": session.session_id,
        "provider": session.provider,
        "transcript_chunks": session.transcript_chunks.len(),
        "transcript_segments": session.transcript_segments.len(),
        "speaker_signals": session.speaker_signals.len(),
        "chat_messages": session.chat_messages.len(),
        "recording_artifacts": list_recording_artifacts(root, &session.session_id),
        "finalization": finalization,
    }))
}

fn preflight_realtime_meeting(root: &Path, args: &[String]) -> Result<Value> {
    let phrase = find_flag_value(args, "--phrase")
        .unwrap_or("Die smarte Tuerklingel verdient ihren Namen und reagiert ohne Verzoegerung.");
    let expected = find_flag_value(args, "--expected").unwrap_or(phrase);
    let foreign_phrase = find_flag_value(args, "--foreign-phrase")
        .unwrap_or("Traditional coding is dead. Inspect driven development is the future.");
    let max_first_delta_ms = find_flag_value(args, "--max-first-delta-ms")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(2500);
    let skip_mistral = args.iter().any(|arg| arg == "--skip-mistral");
    let skip_pulse = args.iter().any(|arg| arg == "--skip-pulse");

    let preflight_id = format!(
        "meeting-realtime-preflight-{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs()
    );
    let dir = root.join("runtime/meeting-preflight").join(&preflight_id);
    fs::create_dir_all(&dir)?;

    let runtime = crate::communication::gateway::runtime_settings_from_root(
        root,
        crate::communication::gateway::CommunicationAdapterKind::Meeting,
    );
    let config = MeetingSessionConfig::from_runtime(
        root,
        "https://teams.microsoft.com/meet/preflight",
        &runtime,
    )?;

    let mut gates = Vec::new();
    let mut artifacts = BTreeMap::new();

    let script_gate = meeting_realtime_script_gate(root);
    gates.push(script_gate.clone());

    let speaker_gate = meeting_speaker_extraction_gate(root);
    gates.push(speaker_gate.clone());

    let mut mistral_gate = json!({
        "name": "mistral_realtime_stream",
        "ok": false,
        "skipped": skip_mistral,
        "reason": if skip_mistral { "skipped_by_flag" } else { "" },
    });
    let mut mistral_probe: Option<Value> = None;
    if !skip_mistral {
        let input_audio = if let Some(path) = find_flag_value(args, "--audio") {
            PathBuf::from(path)
        } else {
            generate_preflight_speech_audio(&dir, "phrase", phrase)?
        };
        artifacts.insert(
            "phrase_audio".to_string(),
            input_audio.display().to_string(),
        );
        match run_mistral_realtime_probe(&dir, &input_audio, &config, expected, max_first_delta_ms)
        {
            Ok(probe) => {
                mistral_gate = json!({
                    "name": "mistral_realtime_stream",
                    "ok": probe.get("ok").and_then(Value::as_bool).unwrap_or(false),
                    "first_delta_ms": probe.get("first_delta_ms"),
                    "expected_match_ratio": probe.get("expected_match_ratio"),
                    "delta_count": probe.get("delta_count"),
                    "final_text": probe.get("final_text"),
                });
                mistral_probe = Some(probe);
            }
            Err(err) => {
                mistral_gate = json!({
                    "name": "mistral_realtime_stream",
                    "ok": false,
                    "reason": err.to_string(),
                });
            }
        }
    }
    gates.push(mistral_gate);

    let pulse_phrase_audio = find_flag_value(args, "--audio").map(PathBuf::from);
    let pulse_foreign_audio = find_flag_value(args, "--foreign-audio").map(PathBuf::from);
    let pulse_gate = if skip_pulse {
        json!({"name": "pulseaudio_isolation", "ok": true, "skipped": true, "reason": "skipped_by_flag"})
    } else {
        match run_pulseaudio_isolation_probe(
            &dir,
            phrase,
            foreign_phrase,
            pulse_phrase_audio.as_deref(),
            pulse_foreign_audio.as_deref(),
            expected,
            &config,
            max_first_delta_ms,
            skip_mistral,
        ) {
            Ok(gate) => gate,
            Err(err) => json!({
                "name": "pulseaudio_isolation",
                "ok": false,
                "reason": err.to_string(),
            }),
        }
    };
    gates.push(pulse_gate);

    let ok = gates
        .iter()
        .all(|gate| gate.get("ok").and_then(Value::as_bool).unwrap_or(false));

    let result = json!({
        "ok": ok,
        "preflight_id": preflight_id,
        "artifact_dir": dir,
        "model": config.realtime_stt_model,
        "policy": {
            "teams_captions_allowed": false,
            "batch_stt_allowed_for_live_ui": false,
            "live_test_allowed_only_if_ok": true,
            "max_first_delta_ms": max_first_delta_ms,
        },
        "artifacts": artifacts,
        "mistral_probe": mistral_probe,
        "gates": gates,
    });
    fs::write(
        dir.join("preflight-result.json"),
        serde_json::to_string_pretty(&result)?,
    )?;
    Ok(result)
}

fn meeting_realtime_script_gate(root: &Path) -> Value {
    let config = MeetingSessionConfig {
        root: root.to_path_buf(),
        meeting_url: "https://teams.microsoft.com/meet/preflight".to_string(),
        provider: MeetingProvider::MicrosoftTeams,
        bot_name: "INF Yoda Notetaker".to_string(),
        max_duration_minutes: 5,
        audio_chunk_seconds: 3,
        stt_model: DEFAULT_MEETING_STT_MODEL.to_string(),
        realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
        mistral_api_key: None,
    };
    match build_meeting_runner_script_with_timeout(&config) {
        Ok(script) => {
            let checks = vec![
                (
                    "uses_mistral_realtime",
                    script.contains("client.audio.realtime.transcribe_stream"),
                ),
                (
                    "uses_pcm_16khz",
                    script.contains("AudioFormat(encoding=\"pcm_s16le\", sample_rate=16000)"),
                ),
                (
                    "has_live_overlay_delta",
                    script.contains("__ctoxTranscriptOverlayLive"),
                ),
                (
                    "has_commit_overlay",
                    script.contains("__ctoxTranscriptOverlayCommit"),
                ),
                (
                    "no_teams_caption_enable",
                    !script.contains("await enableTeamsLiveCaptions"),
                ),
                (
                    "no_teams_caption_polling",
                    script.contains("if (provider === \"microsoft\") return;"),
                ),
                (
                    "no_global_default_sink",
                    !script.contains("set-default-sink virtual_output"),
                ),
                (
                    "no_batch_live_segmenter",
                    !script.contains("audioSegmenter") && !script.contains("teams-audio-chunks"),
                ),
            ];
            let ok = checks.iter().all(|(_, passed)| *passed);
            json!({
                "name": "runner_script_contract",
                "ok": ok,
                "checks": checks.into_iter().map(|(name, ok)| json!({"name": name, "ok": ok})).collect::<Vec<_>>(),
            })
        }
        Err(err) => json!({
            "name": "runner_script_contract",
            "ok": false,
            "reason": err.to_string(),
        }),
    }
}

fn meeting_speaker_extraction_gate(root: &Path) -> Value {
    let config = MeetingSessionConfig {
        root: root.to_path_buf(),
        meeting_url: "https://teams.microsoft.com/meet/preflight".to_string(),
        provider: MeetingProvider::MicrosoftTeams,
        bot_name: "INF Yoda Notetaker".to_string(),
        max_duration_minutes: 5,
        audio_chunk_seconds: 3,
        stt_model: DEFAULT_MEETING_STT_MODEL.to_string(),
        realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
        mistral_api_key: None,
    };
    let mut session = MeetingSession::new(&config);
    let speaker = SpeakerSignal {
        timestamp: now_iso_string(),
        speaker_display: "Michael Welsch".to_string(),
        speaker_id: Some("fixture-speaker".to_string()),
        source: "platform_active_speaker".to_string(),
        confidence: 0.75,
    };
    session.push_stt_transcript(
        "Das ist ein deutscher Realtime Preflight Satz.".to_string(),
        Some(&speaker),
    );
    let segment = session.transcript_segments.first();
    let state_gate = segment
        .map(|segment| {
            segment.speaker_display == "Michael Welsch"
                && segment.source == "stt_with_active_speaker"
                && segment.confidence >= 0.6
        })
        .unwrap_or(false);
    let script = build_meeting_runner_script_with_timeout(&config).unwrap_or_default();
    let dom_gate = script.contains("platform_active_speaker")
        && script.contains("platform_single_participant")
        && script.contains("data-participant-name")
        && script.contains("In dieser Besprechung");
    json!({
        "name": "speaker_attribution_contract",
        "ok": state_gate && dom_gate,
        "checks": [
            {"name": "stt_segments_accept_active_speaker", "ok": state_gate},
            {"name": "teams_dom_speaker_selectors_present", "ok": dom_gate},
        ],
        "note": "Mistral realtime STT has no diarization here; Teams speaker attribution must come from direct Teams DOM state.",
    })
}

fn run_pulseaudio_isolation_probe(
    dir: &Path,
    phrase: &str,
    foreign_phrase: &str,
    phrase_audio_override: Option<&Path>,
    foreign_audio_override: Option<&Path>,
    expected: &str,
    config: &MeetingSessionConfig,
    max_first_delta_ms: u64,
    skip_mistral: bool,
) -> Result<Value> {
    if !cfg!(target_os = "linux") {
        return Ok(json!({
            "name": "pulseaudio_isolation",
            "ok": true,
            "skipped": true,
            "reason": "not_linux",
        }));
    }
    for executable in ["pactl", "ffmpeg"] {
        if !command_available(executable) {
            bail!("{executable} is required for PulseAudio isolation preflight");
        }
    }
    let sink = ensure_virtual_output_sink()?;
    let default_sink = command_stdout("pactl", &["get-default-sink"])
        .unwrap_or_default()
        .trim()
        .to_string();
    if default_sink == "virtual_output" {
        return Ok(json!({
            "name": "pulseaudio_isolation",
            "ok": false,
            "reason": "default_sink_is_virtual_output",
            "detail": "Teams audio may be routed into virtual_output with PULSE_SINK, but CTOX must not make virtual_output the global default sink because that captures unrelated system audio.",
        }));
    }

    let phrase_audio = if let Some(path) = phrase_audio_override {
        path.to_path_buf()
    } else {
        generate_preflight_speech_audio(dir, "pulse_phrase", phrase)?
    };
    let foreign_audio = if let Some(path) = foreign_audio_override {
        path.to_path_buf()
    } else {
        generate_preflight_speech_audio(dir, "pulse_foreign", foreign_phrase)?
    };
    let capture = dir.join("pulse-isolation-capture.wav");
    let mut recorder = Command::new("ffmpeg")
        .args([
            "-y",
            "-loglevel",
            "error",
            "-f",
            "pulse",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-i",
            "virtual_output.monitor",
            "-t",
            "8",
            "-acodec",
            "pcm_s16le",
        ])
        .arg(&capture)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .spawn()
        .context("failed to start PulseAudio isolation recorder")?;
    std::thread::sleep(StdDuration::from_millis(600));
    let mut phrase_player = spawn_audio_player_to_sink(&phrase_audio, "virtual_output")?;
    let mut foreign_player = if default_sink.is_empty() {
        None
    } else {
        Some(spawn_audio_player_to_sink(&foreign_audio, &default_sink)?)
    };
    let _ = phrase_player.wait();
    if let Some(child) = foreign_player.as_mut() {
        let _ = child.wait();
    }
    let recorder_status = recorder.wait()?;
    if !recorder_status.success() {
        bail!("PulseAudio isolation recorder failed with status {recorder_status}");
    }
    if skip_mistral {
        return Ok(json!({
            "name": "pulseaudio_isolation",
            "ok": true,
            "skipped_stt": true,
            "sink": sink,
            "default_sink": default_sink,
            "capture": capture,
        }));
    }
    let probe = run_mistral_realtime_probe(dir, &capture, config, expected, max_first_delta_ms)?;
    let final_text = probe
        .get("final_text")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string();
    let foreign_leak = token_match_ratio(&final_text, foreign_phrase) >= 0.25;
    let ok = probe.get("ok").and_then(Value::as_bool).unwrap_or(false) && !foreign_leak;
    Ok(json!({
        "name": "pulseaudio_isolation",
        "ok": ok,
        "sink": sink,
        "default_sink": default_sink,
        "capture": capture,
        "foreign_leak_detected": foreign_leak,
        "probe": probe,
    }))
}

fn ensure_virtual_output_sink() -> Result<Value> {
    let sources = command_stdout("pactl", &["list", "sources", "short"]).unwrap_or_default();
    if sources.contains("virtual_output.monitor") {
        return Ok(json!({"created": false, "source": "virtual_output.monitor"}));
    }
    let _ = Command::new("pulseaudio")
        .args(["-D", "--exit-idle-time=-1", "--log-level=warning"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status();
    let status = Command::new("pactl")
        .args([
            "load-module",
            "module-null-sink",
            "sink_name=virtual_output",
            "sink_properties=device.description=Virtual_Output",
        ])
        .status()
        .context("failed to create PulseAudio virtual_output sink")?;
    if !status.success() {
        bail!("pactl could not create virtual_output sink");
    }
    let sources = command_stdout("pactl", &["list", "sources", "short"]).unwrap_or_default();
    if !sources.contains("virtual_output.monitor") {
        bail!("virtual_output.monitor is still unavailable after sink creation");
    }
    Ok(json!({"created": true, "source": "virtual_output.monitor"}))
}

fn generate_preflight_speech_audio(dir: &Path, stem: &str, phrase: &str) -> Result<PathBuf> {
    fs::create_dir_all(dir)?;
    let wav = dir.join(format!("{stem}.wav"));
    if command_available("say") {
        let aiff = dir.join(format!("{stem}.aiff"));
        let status = Command::new("say")
            .args(["-v", "Anna", "-o"])
            .arg(&aiff)
            .arg(phrase)
            .status();
        if status.map(|s| s.success()).unwrap_or(false) {
            convert_audio_to_wav(&aiff, &wav)?;
            return Ok(wav);
        }
    }
    for candidate in ["espeak-ng", "espeak"] {
        if !command_available(candidate) {
            continue;
        }
        let status = Command::new(candidate)
            .args(["-v", "de", "-w"])
            .arg(&wav)
            .arg(phrase)
            .status();
        if status.map(|s| s.success()).unwrap_or(false) {
            return Ok(wav);
        }
    }
    bail!(
        "could not generate German speech fixture; install `say`, `espeak-ng`, or pass --audio <wav>"
    )
}

fn convert_audio_to_wav(input: &Path, output: &Path) -> Result<()> {
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(input)
        .args(["-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le"])
        .arg(output)
        .status()
        .context("failed to run ffmpeg audio conversion")?;
    if !status.success() {
        bail!("ffmpeg audio conversion failed for {}", input.display());
    }
    Ok(())
}

fn spawn_audio_player_to_sink(audio: &Path, sink: &str) -> Result<std::process::Child> {
    if command_available("paplay") {
        return Command::new("paplay")
            .arg(format!("--device={sink}"))
            .arg(audio)
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .with_context(|| format!("failed to play {} to sink {sink}", audio.display()));
    }
    Command::new("ffmpeg")
        .args(["-re", "-loglevel", "error", "-i"])
        .arg(audio)
        .args(["-f", "pulse"])
        .arg(sink)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .with_context(|| format!("failed to play {} to sink {sink}", audio.display()))
}

fn run_mistral_realtime_probe(
    dir: &Path,
    audio: &Path,
    config: &MeetingSessionConfig,
    expected: &str,
    max_first_delta_ms: u64,
) -> Result<Value> {
    let api_key = config
        .mistral_api_key
        .as_deref()
        .context("missing CTOX_MISTRAL_API_KEY/MISTRAL_API_KEY for realtime preflight")?;
    if !command_available("ffmpeg") {
        bail!("ffmpeg is required for realtime preflight");
    }
    let pcm = dir.join(format!(
        "{}.pcm",
        audio
            .file_stem()
            .and_then(|value| value.to_str())
            .unwrap_or("preflight-audio")
    ));
    let status = Command::new("ffmpeg")
        .args(["-y", "-loglevel", "error", "-i"])
        .arg(audio)
        .args([
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
        ])
        .arg(&pcm)
        .status()
        .context("failed to convert audio to realtime PCM")?;
    if !status.success() {
        bail!("ffmpeg could not convert {} to PCM", audio.display());
    }
    let script = dir.join("mistral_realtime_preflight.py");
    fs::write(&script, mistral_realtime_preflight_python())?;
    let start = Instant::now();
    let output = Command::new("python3")
        .arg(&script)
        .arg(&pcm)
        .env("CTOX_MISTRAL_API_KEY", api_key)
        .env("MISTRAL_API_KEY", api_key)
        .env(
            "CTOX_MISTRAL_REALTIME_STT_MODEL",
            &config.realtime_stt_model,
        )
        .env("CTOX_MISTRAL_REALTIME_DELAY_MS", "1800")
        .env("CTOX_MISTRAL_REALTIME_PCM_CHUNK_BYTES", "4096")
        .output()
        .context("failed to run Mistral realtime probe")?;
    let elapsed_ms = start.elapsed().as_millis() as u64;
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    fs::write(dir.join("mistral-realtime-probe.jsonl"), &stdout)?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        bail!("Mistral realtime probe failed: {stderr}");
    }
    let mut first_delta_ms = None;
    let mut deltas = Vec::new();
    let mut final_text = String::new();
    for line in stdout.lines() {
        let Ok(value) = serde_json::from_str::<Value>(line) else {
            continue;
        };
        if value.get("type").and_then(Value::as_str) == Some("delta") {
            if first_delta_ms.is_none() {
                first_delta_ms = value.get("ms").and_then(Value::as_u64);
            }
            if let Some(text) = value.get("text").and_then(Value::as_str) {
                deltas.push(text.to_string());
            }
        }
        if value.get("type").and_then(Value::as_str) == Some("summary") {
            final_text = value
                .get("final_text")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
        }
    }
    if final_text.is_empty() {
        final_text = merge_text_deltas(&deltas);
    }
    let match_ratio = token_match_ratio(&final_text, expected);
    let first_delta_ms = first_delta_ms.unwrap_or(u64::MAX);
    let ok = !deltas.is_empty() && first_delta_ms <= max_first_delta_ms && match_ratio >= 0.45;
    Ok(json!({
        "ok": ok,
        "audio": audio,
        "pcm": pcm,
        "elapsed_ms": elapsed_ms,
        "first_delta_ms": if first_delta_ms == u64::MAX { Value::Null } else { json!(first_delta_ms) },
        "max_first_delta_ms": max_first_delta_ms,
        "delta_count": deltas.len(),
        "expected_match_ratio": match_ratio,
        "final_text": final_text,
    }))
}

fn mistral_realtime_preflight_python() -> &'static str {
    r#"import asyncio
import json
import os
import sys
import time

from mistralai.client import Mistral
from mistralai.client.models import AudioFormat

pcm_path = sys.argv[1]
api_key = os.environ.get("CTOX_MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_KEY")
model = os.environ.get("CTOX_MISTRAL_REALTIME_STT_MODEL", "voxtral-mini-transcribe-realtime-2602")
delay_ms = int(os.environ.get("CTOX_MISTRAL_REALTIME_DELAY_MS", "1800"))
chunk_bytes = int(os.environ.get("CTOX_MISTRAL_REALTIME_PCM_CHUNK_BYTES", "4096"))
client = Mistral(api_key=api_key)
started = time.monotonic()
deltas = []

def event_text(event):
    for attr in ("text", "delta", "transcript"):
        value = getattr(event, attr, None)
        if isinstance(value, str) and value.strip():
            return value
    data = getattr(event, "data", None)
    if isinstance(data, dict):
        for key in ("text", "delta", "transcript"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value
    return ""

def event_type(event):
    value = getattr(event, "type", None)
    return value if isinstance(value, str) else type(event).__name__

async def audio_stream():
    with open(pcm_path, "rb") as handle:
        while True:
            data = handle.read(chunk_bytes)
            if not data:
                break
            yield data
            await asyncio.sleep(max(len(data) / 32000.0, 0.01))

async def main():
    async for event in client.audio.realtime.transcribe_stream(
        audio_stream=audio_stream(),
        model=model,
        audio_format=AudioFormat(encoding="pcm_s16le", sample_rate=16000),
        target_streaming_delay_ms=delay_ms,
    ):
        kind = event_type(event)
        now_ms = int((time.monotonic() - started) * 1000)
        if kind == "session.created":
            print(json.dumps({"type": "ready", "ms": now_ms, "model": model, "delay_ms": delay_ms}), flush=True)
            continue
        text = event_text(event)
        if text:
            deltas.append(text)
            print(json.dumps({"type": "delta", "ms": now_ms, "text": text}, ensure_ascii=False), flush=True)
    print(json.dumps({"type": "summary", "final_text": " ".join(deltas)}, ensure_ascii=False), flush=True)

asyncio.run(main())
"#
}

fn merge_text_deltas(deltas: &[String]) -> String {
    let mut merged = String::new();
    for delta in deltas {
        let previous = compact_for_match(&merged);
        let next = compact_for_match(delta);
        if next.is_empty() {
            continue;
        }
        if previous.is_empty() {
            merged = next;
            continue;
        }
        if next == previous || previous.ends_with(&next) {
            continue;
        }
        if next.starts_with(&previous) {
            merged = next;
            continue;
        }
        merged = format!("{previous} {next}");
    }
    merged
}

fn token_match_ratio(actual: &str, expected: &str) -> f64 {
    let actual = compact_for_match(actual);
    let expected_tokens = compact_for_match(expected)
        .split_whitespace()
        .map(ToOwned::to_owned)
        .filter(|token| token.len() >= 3)
        .collect::<Vec<_>>();
    if expected_tokens.is_empty() {
        return 0.0;
    }
    let matched = expected_tokens
        .iter()
        .filter(|token| actual.contains(token.as_str()))
        .count();
    matched as f64 / expected_tokens.len() as f64
}

fn compact_for_match(value: &str) -> String {
    value
        .to_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn command_available(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn command_stdout(command: &str, args: &[&str]) -> Result<String> {
    let output = Command::new(command).args(args).output()?;
    if !output.status.success() {
        bail!("{command} {:?} failed with {}", args, output.status);
    }
    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

// ---------------------------------------------------------------------------
// Meeting runner — spawns Node.js, reads events, drives STT + chat
// ---------------------------------------------------------------------------

fn runner_exit_end_reason(
    wait_result: std::result::Result<&ExitStatus, &std::io::Error>,
    reported_reason: Option<&str>,
    join_failure_reason: Option<&str>,
) -> String {
    let status = match wait_result {
        Ok(status) => status,
        Err(err) => return format!("runner_wait_failed:{err}"),
    };
    if let Some(reason) = join_failure_reason.filter(|reason| !reason.trim().is_empty()) {
        return format!("join_failed:{reason}");
    }
    if !status.success() {
        return format!("runner_exit_status:{status}");
    }
    reported_reason
        .filter(|reason| !reason.trim().is_empty())
        .unwrap_or("runner_exit_success")
        .to_string()
}

fn record_runner_exit(root: &Path, session: &mut MeetingSession, end_reason: String) -> Result<()> {
    session.status = MeetingSessionStatus::Ended;
    if session.ended_at.is_none() {
        session.ended_at = Some(now_iso_string());
    }
    session.end_reason = Some(end_reason);
    session.save(root)
}

/// Run a meeting session synchronously: spawn Playwright, capture audio,
/// transcribe chunks, handle @CTOX mentions, finalize on meeting end.
pub(crate) fn run_meeting_session(root: &Path, config: &MeetingSessionConfig) -> Result<Value> {
    let mut session = MeetingSession::new(config);
    session.save(root)?;

    // Generate the runner script
    let script = build_meeting_runner_script_with_timeout(config)?;

    // Find the Playwright reference dir — script must live inside it so
    // Node's ESM resolver finds the local node_modules/playwright.
    let reference_dir = root.join("runtime/browser/interactive-reference");
    if !reference_dir.exists() {
        bail!(
            "Playwright reference directory not found at {}. Run `cd {} && npm install` first.",
            reference_dir.display(),
            reference_dir.display()
        );
    }
    let script_path = reference_dir.join(format!(".meeting-{}.mjs", session.session_id));
    fs::write(&script_path, &script)?;
    let command_path =
        meeting_sessions_dir(root).join(format!("{}.commands.jsonl", session.session_id));
    fs::write(&command_path, "")?;
    session.stdin_pipe = Some(command_path.display().to_string());
    session.save(root)?;

    // Find Node.js executable
    let node = find_node_executable()?;

    eprintln!(
        "[meeting] Starting {} session: {}",
        config.provider.as_str(),
        config.meeting_url
    );
    eprintln!("[meeting] Script: {}", script_path.display());

    // Pre-flight: start a transient STT backend when the meeting needs one.
    // If STT was already running, leave it alone. If this call starts it only
    // for the meeting, the guard tears it back down after finalization.
    let mut stt_guard = MeetingSttRuntimeGuard::ensure_for_meeting(&config.root);
    let engine_reachable = check_engine_reachable(&config.root);
    let live_transcription_status = native_stt::live_transcription_status_json(&config.root);
    let local_live_ready = live_transcription_status
        .get("local_enabled_for_live_meetings")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let api_live_ready = config.mistral_api_key.is_some();
    let live_meeting_ready = local_live_ready || api_live_ready;
    session.engine_was_reachable_at_start = engine_reachable;
    session.live_transcription_ready_at_start = live_meeting_ready;
    session.live_transcription_status_at_start = Some(live_transcription_status.clone());
    if engine_reachable {
        eprintln!("[meeting] STT runtime reachable via managed transport");
    } else {
        eprintln!("[meeting] WARNING: STT runtime not reachable via managed transport");
        eprintln!("[meeting] Audio chunks will still be captured and saved to disk.");
        if let Some(reason) = stt_guard.start_error.as_deref() {
            eprintln!("[meeting] STT auto-start failed: {reason}");
        }
        eprintln!(
            "[meeting] Unsent chunks will be retried at meeting end if the engine becomes available."
        );
    }
    if live_meeting_ready {
        if api_live_ready {
            eprintln!("[meeting] Mistral realtime STT enabled for live meeting");
        } else {
            eprintln!("[meeting] local live STT enabled; realtime streaming proof is present");
        }
    } else {
        let reason = live_transcription_status
            .get("local_live_disabled_reason")
            .and_then(Value::as_str)
            .unwrap_or("not_live_ready");
        eprintln!(
            "[meeting] live STT disabled ({reason}); microsoft meeting overlay requires realtime STT and will not fall back to Teams captions"
        );
    }

    // Spawn the Node.js process. On Linux VPS hosts there is usually no
    // interactive X server, but Teams needs a headed browser for media capture.
    let mut runner_cmd = build_meeting_runner_command(&node, &reference_dir, &script_path)?;
    if config.provider == MeetingProvider::MicrosoftTeams && cfg!(target_os = "linux") {
        match ensure_virtual_output_sink() {
            Ok(info) => eprintln!("[meeting] PulseAudio Teams sink ready: {info}"),
            Err(err) => eprintln!("[meeting] WARNING: PulseAudio Teams sink setup failed: {err}"),
        }
        // Route only the Teams browser process tree to the virtual sink. Do not
        // change PulseAudio's global default sink, otherwise unrelated system
        // audio can leak into the meeting transcript.
        runner_cmd.env("PULSE_SINK", "virtual_output");
    }
    runner_cmd
        .env("CTOX_MEETING_COMMAND_FILE", &command_path)
        .env(
            "CTOX_MISTRAL_REALTIME_STT_MODEL",
            &config.realtime_stt_model,
        )
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(api_key) = config.mistral_api_key.as_deref() {
        runner_cmd
            .env("CTOX_MISTRAL_API_KEY", api_key)
            .env("MISTRAL_API_KEY", api_key);
    }
    let mut child = runner_cmd.spawn().with_context(|| {
        format!(
            "failed to spawn meeting browser runner via {:?}",
            runner_cmd
        )
    })?;

    // Drain stderr in a background thread so we surface Node.js errors
    // (otherwise the pipe fills, blocks, and we never see the failure)
    if let Some(stderr) = child.stderr.take() {
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines().map_while(Result::ok) {
                eprintln!("[meeting:node] {line}");
            }
        });
    }

    session.pid = Some(child.id());
    session.status = MeetingSessionStatus::Running;
    session.save(root)?;

    let stdout = child.stdout.take().context("no stdout from node process")?;
    let stdin = child.stdin.take();

    // Read stdout line by line (JSON-lines protocol)
    let reader = BufReader::new(stdout);
    let mut join_failure_reason: Option<String> = None;
    let mut runner_reported_end_reason: Option<String> = None;
    let mut last_speaker_signal: Option<SpeakerSignal> = None;
    let speaker_probe_path =
        meeting_sessions_dir(root).join(format!("{}-speaker-probes.jsonl", session.session_id));
    for line in reader.lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let event: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => {
                eprintln!(
                    "[meeting] non-JSON output: {}",
                    &line[..line.len().min(200)]
                );
                continue;
            }
        };

        let event_type = event.get("type").and_then(Value::as_str).unwrap_or("");

        match event_type {
            "status" => {
                let status = event.get("status").and_then(Value::as_str).unwrap_or("");
                eprintln!("[meeting] status: {status}");
            }
            "joined" => {
                let reason = event.get("reason").and_then(Value::as_str).unwrap_or("");
                if reason.is_empty() {
                    eprintln!("[meeting] Joined meeting successfully");
                } else {
                    eprintln!("[meeting] Joined meeting successfully ({reason})");
                }
                session.status = MeetingSessionStatus::Active;
                session.save(root)?;
            }
            "join_failed" => {
                let reason = event.get("reason").and_then(Value::as_str).unwrap_or("");
                eprintln!("[meeting] join verification failed: {reason}");
                session.status = MeetingSessionStatus::JoinFailed;
                session.ended_at = Some(now_iso_string());
                session.save(root)?;
                join_failure_reason = Some(reason.to_string());
            }
            "audio_chunk" => {
                let chunk_path = event.get("path").and_then(Value::as_str).unwrap_or("");
                if chunk_path.is_empty() {
                    continue;
                }
                // Copy the chunk into the session's persistent directory so it
                // survives after the node process exits (the JS writes to a
                // tempDir that gets cleaned up).
                let persisted_path = persist_audio_chunk(root, &session.session_id, chunk_path);
                let chunk_for_stt = persisted_path.as_deref().unwrap_or(chunk_path);

                match transcribe_audio_chunk(
                    &config.root,
                    Path::new(chunk_for_stt),
                    &config.stt_model,
                ) {
                    Ok(text) if !text.is_empty() => {
                        eprintln!("[meeting] transcript: {}...", &text[..text.len().min(80)]);
                        let direct_speaker = recent_direct_speaker(last_speaker_signal.as_ref());
                        session.push_stt_transcript(text, direct_speaker);
                        session.save(root)?;
                        if let Some(p) = persisted_path.as_ref() {
                            let _ = fs::remove_file(p);
                        }
                    }
                    Ok(_) => {
                        // Empty transcript (silence) — drop the chunk, it's useless
                        if let Some(p) = persisted_path.as_ref() {
                            let _ = fs::remove_file(p);
                        }
                    }
                    Err(err) => {
                        eprintln!("[meeting] STT error: {err}");
                        // Keep the chunk for retry at finalize time
                        if let Some(p) = persisted_path {
                            session.pending_audio_chunks.push(p);
                            session.save(root)?;
                        }
                    }
                }
            }
            "active_speaker" => {
                if let Some(signal) = SpeakerSignal::from_event(&event) {
                    eprintln!(
                        "[meeting] active speaker [{}]: {}",
                        signal.source, signal.speaker_display
                    );
                    last_speaker_signal = Some(signal.clone());
                    session.speaker_signals.push(signal);
                    session.save(root)?;
                }
            }
            "speaker_probe" => {
                let text = event.get("text").and_then(Value::as_str).unwrap_or("");
                eprintln!("[meeting] speaker probe: {}", &text[..text.len().min(500)]);
                if !text.trim().is_empty() {
                    if let Ok(encoded) = serde_json::to_string(&event) {
                        let _ = append_line(&speaker_probe_path, &encoded);
                    }
                }
            }
            "transcript_segment" => {
                if let Some(segment) = TranscriptSegment::from_platform_event(&event) {
                    eprintln!(
                        "[meeting] transcript segment [{}] {}: {}...",
                        segment.source,
                        segment.speaker_display,
                        &segment.text[..segment.text.len().min(80)]
                    );
                    session.push_platform_transcript(segment);
                    session.save(root)?;
                }
            }
            "chat" => {
                let sender = event
                    .get("sender")
                    .and_then(Value::as_str)
                    .unwrap_or("Unknown");
                let text = event.get("text").and_then(Value::as_str).unwrap_or("");
                let ts = event.get("ts").and_then(Value::as_str).unwrap_or("");

                // --- Self-loop protection ---
                // Skip messages that originated from this bot itself, or that
                // duplicate the most recent message we sent (sometimes the
                // chat-send round-trips back through the scraper).
                if session.is_own_message(sender, text) {
                    eprintln!(
                        "[meeting] skipped own message: {}",
                        &text[..text.len().min(60)]
                    );
                    continue;
                }

                eprintln!("[meeting] chat [{sender}]: {text}");
                session.chat_messages.push(ChatMessage {
                    sender: sender.to_string(),
                    text: text.to_string(),
                    timestamp: ts.to_string(),
                });
                if MeetingSession::is_mention(text) {
                    let ack_text = first_mention_ack_text();
                    if !session
                        .outbound_chat_texts
                        .iter()
                        .any(|sent| normalize_chat_text(sent) == normalize_chat_text(ack_text))
                    {
                        match write_chat_command_to_session(&session.to_json(), ack_text, None) {
                            Ok(()) => {
                                eprintln!("[meeting] queued immediate mention ack");
                                session.outbound_chat_texts.push(ack_text.to_string());
                            }
                            Err(err) => {
                                eprintln!(
                                    "[meeting] warning: could not queue immediate mention ack: {err}"
                                );
                            }
                        }
                    }
                }
                // Persist session so sync() can pick up new chat messages
                session.save(root)?;

                // @CTOX mentions are now ingested as normal inbound messages
                // via sync() → upsert_communication_message(). The service
                // loop's route_external_messages() will pick them up and
                // route them to the agent with the meeting-participant skill.
                // No extra queue task needed — the standard pipeline handles it.
            }
            "command_received" => {
                let action = event.get("action").and_then(Value::as_str).unwrap_or("");
                eprintln!("[meeting] command received: {action}");
            }
            "chat_sent" => {
                let text = event.get("text").and_then(Value::as_str).unwrap_or("");
                let message_key = event
                    .get("message_key")
                    .and_then(Value::as_str)
                    .unwrap_or("");
                eprintln!("[meeting] chat sent: {}", &text[..text.len().min(80)]);
                if !message_key.is_empty() {
                    match confirm_meeting_outbound_message(root, &session.session_id, message_key) {
                        Ok(true) => eprintln!(
                            "[meeting] outbound message confirmed sent: {message_key}"
                        ),
                        Ok(false) => eprintln!(
                            "[meeting] warning: no submitted outbound row found for confirmation {message_key}"
                        ),
                        Err(err) => eprintln!(
                            "[meeting] warning: could not persist sent confirmation for {message_key}: {err}"
                        ),
                    }
                }
                if !text.is_empty() {
                    session.outbound_chat_texts.push(text.to_string());
                    session.save(root)?;
                }
            }
            "chat_send_failed" => {
                let text = event.get("text").and_then(Value::as_str).unwrap_or("");
                eprintln!(
                    "[meeting] chat send failed: {}",
                    &text[..text.len().min(80)]
                );
            }
            "recording_artifact" => {
                let artifact_path = event.get("path").and_then(Value::as_str).unwrap_or("");
                if artifact_path.is_empty() {
                    continue;
                }
                match persist_recording_artifact(root, &session.session_id, artifact_path) {
                    Some(path) => eprintln!("[meeting] recording artifact: {path}"),
                    None => eprintln!(
                        "[meeting] recording artifact persist failed: {}",
                        &artifact_path[..artifact_path.len().min(160)]
                    ),
                }
            }
            "ffmpeg_error" => {
                let text = event.get("text").and_then(Value::as_str).unwrap_or("");
                eprintln!("[meeting] ffmpeg error: {}", &text[..text.len().min(200)]);
            }
            "ffmpeg_exit" => {
                let code = event.get("code").and_then(Value::as_i64).unwrap_or(-1);
                eprintln!("[meeting] ffmpeg exited with code {code}");
            }
            "participant_count" => {
                let count = event.get("count").and_then(Value::as_u64).unwrap_or(0);
                eprintln!("[meeting] participants: {count}");
            }
            "ended" => {
                let reason = event
                    .get("reason")
                    .and_then(Value::as_str)
                    .unwrap_or("unknown");
                eprintln!("[meeting] Meeting ended: {reason}");
                runner_reported_end_reason = Some(reason.to_string());
            }
            "finalized" => {
                break;
            }
            "error" => {
                let msg = event.get("message").and_then(Value::as_str).unwrap_or("");
                eprintln!("[meeting] error: {msg}");
            }
            "warning" => {
                let msg = event.get("message").and_then(Value::as_str).unwrap_or("");
                eprintln!("[meeting] warning: {msg}");
            }
            "browser_log" => {
                let level = event.get("level").and_then(Value::as_str).unwrap_or("log");
                let text = event.get("text").and_then(Value::as_str).unwrap_or("");
                eprintln!("[meeting:browser:{level}] {text}");
            }
            _ => {}
        }
    }

    // The process monitor owns terminal session persistence. This covers normal
    // finalization and runner crashes that close stdout without an `ended` event.
    let wait_result = child.wait();
    let end_reason = runner_exit_end_reason(
        wait_result.as_ref(),
        runner_reported_end_reason.as_deref(),
        join_failure_reason.as_deref(),
    );
    record_runner_exit(root, &mut session, end_reason)?;

    // Clean up script file
    let _ = fs::remove_file(&script_path);

    if let Some(reason) = join_failure_reason {
        drop(stdin); // close stdin pipe
        stt_guard.finish();
        return Ok(json!({
            "ok": false,
            "session_id": session.session_id,
            "provider": session.provider,
            "status": "join_failed",
            "reason": reason,
            "transcript_chunks": session.transcript_chunks.len(),
            "transcript_segments": session.transcript_segments.len(),
            "speaker_signals": session.speaker_signals.len(),
            "chat_messages": session.chat_messages.len(),
            "pending_audio_chunks": session.pending_audio_chunks.len(),
            "finalization": {
                "action": "skipped",
                "reason": "meeting was not joined"
            },
        }));
    }

    // Finalize
    session.status = MeetingSessionStatus::Ended;
    if session.ended_at.is_none() {
        session.ended_at = Some(now_iso_string());
    }
    session.save(root)?;

    // Lazy re-transcription: if the engine is now reachable and we have
    // pending chunks from failed STT attempts, retry them now.
    let retry_result = retry_pending_audio_chunks(root, &mut session, config);
    let recording_transcript_result =
        transcribe_full_recording_if_needed(root, &mut session, config);

    let finalization = finalize_meeting(root, &session, config)?;

    drop(stdin); // close stdin pipe
    stt_guard.finish();

    Ok(json!({
        "ok": true,
        "session_id": session.session_id,
        "provider": session.provider,
        "status": "finalized",
        "transcript_chunks": session.transcript_chunks.len(),
        "transcript_segments": session.transcript_segments.len(),
        "speaker_signals": session.speaker_signals.len(),
        "chat_messages": session.chat_messages.len(),
        "pending_audio_chunks": session.pending_audio_chunks.len(),
        "stt_retry": retry_result,
        "recording_transcript": recording_transcript_result,
        "finalization": finalization,
    }))
}

/// Finalize a meeting: combine transcript, create system-onboarding queue task.
fn finalize_meeting(
    root: &Path,
    session: &MeetingSession,
    _config: &MeetingSessionConfig,
) -> Result<Value> {
    let transcript = session.full_transcript();
    let chat_log = session.full_chat_log();

    if transcript.is_empty() && chat_log.is_empty() {
        return Ok(json!({
            "action": "skipped",
            "reason": "no transcript or chat content to process",
        }));
    }

    // Save full transcript to file
    let transcript_path =
        meeting_sessions_dir(root).join(format!("{}-transcript.txt", session.session_id));
    fs::write(&transcript_path, &transcript)?;

    let chat_log_path =
        meeting_sessions_dir(root).join(format!("{}-chatlog.txt", session.session_id));
    if !chat_log.is_empty() {
        fs::write(&chat_log_path, &chat_log)?;
    }
    let recording_artifacts = list_recording_artifacts(root, &session.session_id);
    let artifact_manifest_path =
        meeting_sessions_dir(root).join(format!("{}-artifacts.json", session.session_id));
    fs::write(
        &artifact_manifest_path,
        serde_json::to_string_pretty(&json!({
            "session_id": session.session_id,
            "provider": session.provider,
            "meeting_url": session.meeting_url,
            "started_at": session.started_at,
            "ended_at": session.ended_at,
            "transcript_path": transcript_path.display().to_string(),
            "chatlog_path": chat_log_path.display().to_string(),
            "transcript_segment_count": session.transcript_segments.len(),
            "speaker_signal_count": session.speaker_signals.len(),
            "recording_artifacts": recording_artifacts,
        }))?,
    )?;

    // Build the post-meeting processing prompt
    let prompt = format!(
        "## Post-meeting transcript processing\n\
         \n\
         A **{provider}** meeting has ended. Process the transcript and chat log below.\n\
         \n\
         ### Meeting metadata\n\
         - Provider: {provider}\n\
         - URL: {url}\n\
         - Session: `{session_id}`\n\
         - Started: {started}\n\
         - Ended: {ended}\n\
         - Total transcript chunks: {chunk_count}\n\
         - Total structured transcript segments: {segment_count}\n\
         - Total speaker signals: {speaker_signal_count}\n\
         - Total chat messages: {chat_count}\n\
         - Transcript file: `{transcript_path}`\n\
         \n\
         ### What to extract\n\
         \n\
         Read the transcript carefully and extract:\n\
         \n\
         1. **Decisions** -- What was agreed upon? By whom?\n\
         2. **Action items** -- Who committed to doing what? By when?\n\
         3. **Open questions** -- What was discussed but not resolved?\n\
         4. **Reusable operational knowledge candidates** -- Only extract items that can become a \
         durable Skillbook/Runbook/Runbook-Item. Meeting facts, status notes, and one-off decisions are \
         not knowledge by themselves; keep those in the summary or tickets.\n\
         \n\
         ### What to create\n\
         \n\
         - For each **action item**: Create a ticket with clear title, assignee, and deadline.\n\
         - For each **reusable operational knowledge candidate**: create or update a Skillbook/Runbook \
         bundle via `ctox ticket source-skill-import-bundle`. The durable knowledge artifact must land in \
         `knowledge_main_skills`, `knowledge_skillbooks`, `knowledge_runbooks`, and \
         `knowledge_runbook_items`. Do not use `ticket_knowledge_entries` as the final knowledge store.\n\
         - If the meeting produced only facts, decisions, or follow-up work and no reusable procedure, \
         do not create a knowledge artifact; keep the facts in the meeting summary and tickets.\n\
         - For **open questions**: Create a follow-up queue task.\n\
         - **Always**: Send a meeting summary to the relevant communication channel.\n\
         \n\
         ### Structured extraction contract\n\
         \n\
         Start by producing a compact JSON object with keys `decisions`, `action_items`, \
         `open_questions`, `runbook_candidates`, and `tickets_to_create`. Then perform the durable writes. \
         Each `runbook_candidates` item must name the target skillbook/runbook and explain why it is \
         reusable operational procedure rather than a meeting note. \
         Every ticket candidate must include `title`, `body`, `source_session_id`, and \
         `dedupe_rationale`.\n\
         \n\
         ### Quality checks\n\
         \n\
         - Use a participant name only when a transcript line has a platform speaker source or high confidence.\n\
         - If the source is plain STT, unknown, or low-confidence active-speaker correlation, say \"a participant\" instead of inventing a name.\n\
         - Distinguish between decisions (confirmed) and suggestions (discussed but not confirmed).\n\
         - Check existing tickets before creating duplicates.\n\
         - The summary should be something a human who missed the meeting can act on.\n\
         \n\
         ### Full transcript\n\
         {transcript}\n\
         \n\
         ### Chat log\n\
         {chat_log}\n",
        provider = session.provider,
        url = session.meeting_url,
        session_id = session.session_id,
        started = session.started_at,
        ended = session.ended_at.as_deref().unwrap_or("unknown"),
        chunk_count = session.transcript_chunks.len(),
        segment_count = session.transcript_segments.len(),
        speaker_signal_count = session.speaker_signals.len(),
        chat_count = session.chat_messages.len(),
        transcript_path = transcript_path.display(),
        transcript = if transcript.is_empty() {
            "(empty)"
        } else {
            &transcript
        },
        chat_log = if chat_log.is_empty() {
            "(no chat)"
        } else {
            &chat_log
        },
    );

    let post_meeting_ticket = crate::mission::ticket_local_native::create_local_ticket(
        root,
        &format!("Meeting Nachbereitung: {}", session.provider),
        &format!(
            "Default post-meeting processing ticket for session `{}`.\n\nTranscript: {}\nChat log: {}\nArtifact manifest: {}",
            session.session_id,
            transcript_path.display(),
            chat_log_path.display(),
            artifact_manifest_path.display(),
        ),
        Some("open"),
        Some("normal"),
    )
    .ok();

    // Ingest the summary as a normal inbound message in the "meeting" channel.
    // The service loop's route_external_messages() will pick it up and route
    // it to the agent with the meeting-participant skill via metadata.
    let db_path = root.join("runtime/ctox.sqlite3");
    let mut conn = open_channel_db(&db_path)?;
    let observed_at = now_iso_string();
    let message_key = format!(
        "meeting::{}::summary::{}",
        session.session_id,
        stable_digest(&format!("summary:{}", observed_at))
    );
    let metadata = json!({
        "provider": session.provider,
        "session_id": session.session_id,
        "source": "meeting_summary",
        "skill": "meeting-participant",
        "transcript_path": transcript_path.display().to_string(),
        "artifact_manifest_path": artifact_manifest_path.display().to_string(),
        "recording_artifacts": recording_artifacts.clone(),
        "transcript_segment_count": session.transcript_segments.len(),
        "speaker_signal_count": session.speaker_signals.len(),
        "post_meeting_ticket_id": post_meeting_ticket.as_ref().map(|ticket| ticket.ticket_id.clone()),
    });
    upsert_communication_message(
        &mut conn,
        UpsertMessage {
            message_key: &message_key,
            channel: "meeting",
            account_key: "meeting:system",
            thread_key: &session.session_id,
            remote_id: &message_key,
            direction: "inbound",
            folder_hint: "summary",
            sender_display: "Meeting Bot",
            sender_address: "ctox@local",
            recipient_addresses_json: "[]",
            cc_addresses_json: "[]",
            bcc_addresses_json: "[]",
            subject: &format!("{} meeting summary", session.provider),
            preview: &format!(
                "{} meeting ended — {} transcript chunks, {} chat messages",
                session.provider,
                session.transcript_chunks.len(),
                session.chat_messages.len()
            ),
            body_text: &prompt,
            body_html: "",
            raw_payload_ref: "",
            trust_level: "internal",
            status: "received",
            seen: false,
            has_attachments: false,
            external_created_at: &observed_at,
            observed_at: &observed_at,
            metadata_json: &serde_json::to_string(&metadata)?,
        },
    )?;
    refresh_thread(&mut conn, &session.session_id)?;
    ensure_routing_rows_for_inbound(&conn)?;

    Ok(json!({
        "action": "ingested",
        "message_key": message_key,
        "transcript_path": transcript_path.display().to_string(),
        "artifact_manifest_path": artifact_manifest_path.display().to_string(),
        "recording_artifact_count": recording_artifacts.len(),
        "post_meeting_ticket_id": post_meeting_ticket.as_ref().map(|ticket| ticket.ticket_id.clone()),
        "skill": "meeting-participant",
    }))
}

fn list_recording_artifacts(root: &Path, session_id: &str) -> Vec<String> {
    let session_dir = meeting_sessions_dir(root);
    let mut artifacts = Vec::new();
    if let Ok(entries) = fs::read_dir(&session_dir) {
        artifacts.extend(
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .map(|name| name.starts_with(session_id))
                        .unwrap_or(false)
                })
                .filter(|path| is_recording_media_path(path))
                .map(|path| path.display().to_string()),
        );
    }

    let artifact_dir = meeting_sessions_dir(root).join(format!("{session_id}-audio"));
    if let Ok(entries) = fs::read_dir(&artifact_dir) {
        artifacts.extend(
            entries
                .filter_map(Result::ok)
                .map(|entry| entry.path())
                .filter(|path| path.is_file())
                .filter(|path| is_recording_media_path(path))
                .map(|path| path.display().to_string()),
        );
    }
    artifacts.sort();
    artifacts.dedup();
    artifacts
}

fn is_recording_media_path(path: &Path) -> bool {
    path.extension()
        .and_then(|ext| ext.to_str())
        .map(|ext| {
            matches!(
                ext.to_ascii_lowercase().as_str(),
                "webm" | "mp4" | "wav" | "m4a" | "ogg"
            )
        })
        .unwrap_or(false)
}

fn is_full_meeting_recording_path(path: &Path, session_id: &str) -> bool {
    path.file_name()
        .and_then(|name| name.to_str())
        .map(|name| {
            name.starts_with(session_id)
                && name.to_ascii_lowercase().contains("recording")
                && is_recording_media_path(path)
        })
        .unwrap_or(false)
}

fn full_meeting_recording_candidates(root: &Path, session_id: &str) -> Vec<PathBuf> {
    let mut candidates = list_recording_artifacts(root, session_id)
        .into_iter()
        .map(PathBuf::from)
        .filter(|path| is_full_meeting_recording_path(path, session_id))
        .collect::<Vec<_>>();
    candidates.sort_by(|left, right| {
        let left_len = fs::metadata(left).map(|meta| meta.len()).unwrap_or(0);
        let right_len = fs::metadata(right).map(|meta| meta.len()).unwrap_or(0);
        right_len.cmp(&left_len).then_with(|| left.cmp(right))
    });
    candidates
}

fn meeting_transcript_needs_recording_fallback(session: &MeetingSession) -> bool {
    let has_stt_segment = session
        .transcript_segments
        .iter()
        .any(|segment| segment.source.starts_with("stt"));
    if has_stt_segment {
        return false;
    }
    session.full_transcript().trim().chars().count() < 2_000
}

fn transcribe_full_recording_if_needed(
    root: &Path,
    session: &mut MeetingSession,
    config: &MeetingSessionConfig,
) -> Value {
    if !meeting_transcript_needs_recording_fallback(session) {
        return json!({"action": "skipped", "reason": "transcript already has usable STT or captions"});
    }

    let candidates = full_meeting_recording_candidates(root, &session.session_id);
    let Some(recording_path) = candidates.first() else {
        return json!({"action": "skipped", "reason": "no full recording artifact"});
    };

    eprintln!(
        "[meeting] transcript is incomplete; transcribing full recording {}",
        recording_path.display()
    );
    match transcribe_audio_chunk(&config.root, recording_path, &config.stt_model) {
        Ok(text) if !text.trim().is_empty() => {
            let text_chars = text.chars().count();
            session.push_stt_transcript(text, None);
            let _ = session.save(root);
            json!({
                "action": "transcribed_full_recording",
                "recording_path": recording_path.display().to_string(),
                "text_chars": text_chars,
            })
        }
        Ok(_) => json!({
            "action": "skipped",
            "reason": "full recording transcription returned empty text",
            "recording_path": recording_path.display().to_string(),
        }),
        Err(err) => {
            eprintln!(
                "[meeting] full recording transcription failed for {}: {err}",
                recording_path.display()
            );
            json!({
                "action": "failed",
                "reason": err.to_string(),
                "recording_path": recording_path.display().to_string(),
            })
        }
    }
}

/// At finalize time, re-check if the STT engine is reachable. If it is and we
/// have pending audio chunks from earlier failures, transcribe them now and
/// append the results to the transcript. Successfully transcribed chunks are
/// removed from disk. Returns a summary value.
fn retry_pending_audio_chunks(
    root: &Path,
    session: &mut MeetingSession,
    config: &MeetingSessionConfig,
) -> Value {
    if session.pending_audio_chunks.is_empty() {
        return json!({"action": "skipped", "reason": "no pending chunks"});
    }
    let engine_now_reachable = check_engine_reachable(&config.root);
    if !engine_now_reachable {
        return json!({
            "action": "skipped",
            "reason": "engine still unreachable",
            "pending_count": session.pending_audio_chunks.len(),
        });
    }

    eprintln!(
        "[meeting] STT engine now reachable — retrying {} pending chunks",
        session.pending_audio_chunks.len()
    );
    let mut succeeded = 0u32;
    let mut still_failing = Vec::new();
    let pending = std::mem::take(&mut session.pending_audio_chunks);
    for chunk_path in pending {
        match transcribe_audio_chunk(&config.root, Path::new(&chunk_path), &config.stt_model) {
            Ok(text) if !text.is_empty() => {
                session.push_stt_transcript(text, None);
                let _ = fs::remove_file(&chunk_path);
                succeeded += 1;
            }
            Ok(_) => {
                // Silence — drop the chunk
                let _ = fs::remove_file(&chunk_path);
            }
            Err(err) => {
                eprintln!("[meeting] retry STT error on {}: {err}", chunk_path);
                still_failing.push(chunk_path);
            }
        }
    }
    session.pending_audio_chunks = still_failing.clone();
    let _ = session.save(root);

    json!({
        "action": "retried",
        "succeeded": succeeded,
        "still_failing": still_failing.len(),
    })
}

/// Copy an audio chunk from the node-managed tempDir into a persistent
/// per-session directory so it survives after the node process exits.
/// Returns the persisted path, or None if the copy failed.
fn persist_audio_chunk(root: &Path, session_id: &str, source_path: &str) -> Option<String> {
    let src = Path::new(source_path);
    if !src.exists() {
        return None;
    }
    let metadata = fs::metadata(src).ok()?;
    if metadata.len() < 4096 {
        return None;
    }
    let dest_dir = meeting_sessions_dir(root).join(format!("{session_id}-audio"));
    if fs::create_dir_all(&dest_dir).is_err() {
        return None;
    }
    let filename = src.file_name()?;
    let dest = dest_dir.join(filename);
    if fs::copy(src, &dest).is_ok() {
        Some(dest.display().to_string())
    } else {
        None
    }
}

fn persist_recording_artifact(root: &Path, session_id: &str, source_path: &str) -> Option<String> {
    let src = Path::new(source_path);
    if !src.exists() {
        return None;
    }
    let ext = src
        .extension()
        .and_then(|value| value.to_str())
        .filter(|value| !value.trim().is_empty())
        .unwrap_or("mp4");
    let dest = meeting_sessions_dir(root).join(format!("{session_id}-recording.{ext}"));
    if fs::copy(src, &dest).is_ok() {
        Some(dest.display().to_string())
    } else {
        None
    }
}

/// Check whether the managed STT runtime responds on its configured transport.
pub(crate) fn check_engine_reachable(root: &Path) -> bool {
    crate::communication::gateway::transcription_backend_reachable(root)
}

fn find_node_executable() -> Result<String> {
    for candidate in ["node", "/usr/local/bin/node", "/usr/bin/node"] {
        if Command::new(candidate)
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .is_ok()
        {
            return Ok(candidate.to_string());
        }
    }
    // Try to find via PATH
    if let Ok(output) = Command::new("which").arg("node").output() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Ok(path);
        }
    }
    bail!("node executable not found — install Node.js >= 18")
}

fn build_meeting_runner_command(
    node: &str,
    reference_dir: &Path,
    script_path: &Path,
) -> Result<Command> {
    if should_wrap_browser_runner_with_xvfb(std::env::var_os("DISPLAY").as_deref()) {
        let xvfb_run = find_xvfb_run_executable().with_context(|| {
            "DISPLAY is not set and xvfb-run was not found; install xvfb for VPS meeting capture"
        })?;
        eprintln!(
            "[meeting] DISPLAY is not set; launching headed browser runner via {}",
            xvfb_run.display()
        );
        let mut cmd = Command::new(xvfb_run);
        cmd.current_dir(reference_dir)
            .arg("-a")
            .arg("-s")
            .arg(MEETING_XVFB_SERVER_ARGS)
            .arg(node)
            .arg(script_path);
        Ok(cmd)
    } else {
        let mut cmd = Command::new(node);
        cmd.current_dir(reference_dir).arg(script_path);
        Ok(cmd)
    }
}

fn should_wrap_browser_runner_with_xvfb(display: Option<&OsStr>) -> bool {
    cfg!(target_os = "linux") && display.map(|value| value.is_empty()).unwrap_or(true)
}

fn find_xvfb_run_executable() -> Option<PathBuf> {
    for candidate in ["/usr/bin/xvfb-run", "/usr/local/bin/xvfb-run"] {
        let path = PathBuf::from(candidate);
        if path.exists() {
            return Some(path);
        }
    }
    if let Ok(output) = Command::new("which").arg("xvfb-run").output() {
        let path = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !path.is_empty() {
            return Some(PathBuf::from(path));
        }
    }
    None
}

// ---------------------------------------------------------------------------
// Meeting provider detection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum MeetingProvider {
    GoogleMeet,
    MicrosoftTeams,
    Zoom,
}

impl MeetingProvider {
    pub(crate) fn detect(url: &str) -> Option<Self> {
        let lower = url.to_lowercase();
        if lower.contains("meet.google.com") {
            Some(Self::GoogleMeet)
        } else if lower.contains("teams.microsoft.com") || lower.contains("teams.live.com") {
            Some(Self::MicrosoftTeams)
        } else if lower.contains("zoom.us") || lower.contains("zoom.com") {
            Some(Self::Zoom)
        } else {
            None
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            Self::GoogleMeet => "google",
            Self::MicrosoftTeams => "microsoft",
            Self::Zoom => "zoom",
        }
    }
}

// ---------------------------------------------------------------------------
// Meeting session management
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub(crate) struct MeetingSessionConfig {
    pub root: PathBuf,
    pub meeting_url: String,
    pub provider: MeetingProvider,
    pub bot_name: String,
    pub max_duration_minutes: u64,
    pub audio_chunk_seconds: u64,
    pub stt_model: String,
    pub realtime_stt_model: String,
    pub mistral_api_key: Option<String>,
}

impl MeetingSessionConfig {
    pub(crate) fn from_runtime(
        root: &Path,
        meeting_url: &str,
        runtime: &BTreeMap<String, String>,
    ) -> Result<Self> {
        let provider = MeetingProvider::detect(meeting_url)
            .context("cannot detect meeting provider from URL")?;
        let bot_name = runtime_setting_or_env(runtime, "CTO_MEETING_BOT_NAME")
            .unwrap_or_else(|| "INF Yoda Notetaker".to_string());
        let max_duration_minutes =
            runtime_setting_or_env(runtime, "CTO_MEETING_MAX_DURATION_MINUTES")
                .and_then(|v| v.parse().ok())
                .unwrap_or(180u64);
        let audio_chunk_seconds =
            runtime_setting_or_env(runtime, "CTO_MEETING_AUDIO_CHUNK_SECONDS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(3u64);
        let stt_model = normalize_meeting_stt_model(
            runtime_setting_or_env(runtime, "CTOX_STT_MODEL").as_deref(),
        );
        let realtime_stt_model = runtime_setting_or_env(runtime, "CTOX_MISTRAL_REALTIME_STT_MODEL")
            .or_else(|| runtime_setting_or_env(runtime, "CTOX_STT_REALTIME_MODEL"))
            .unwrap_or_else(|| "voxtral-mini-transcribe-realtime-2602".to_string());
        let mistral_api_key = runtime_setting_or_env(runtime, "CTOX_MISTRAL_API_KEY")
            .or_else(|| runtime_setting_or_env(runtime, "MISTRAL_API_KEY"))
            .or_else(|| crate::secrets::get_credential(root, "CTOX_MISTRAL_API_KEY"))
            .or_else(|| crate::secrets::get_credential(root, "MISTRAL_API_KEY"));
        Ok(Self {
            root: root.to_path_buf(),
            meeting_url: meeting_url.to_string(),
            provider,
            bot_name,
            max_duration_minutes,
            audio_chunk_seconds,
            stt_model,
            realtime_stt_model,
            mistral_api_key,
        })
    }
}

fn runtime_setting_or_env(runtime: &BTreeMap<String, String>, key: &str) -> Option<String> {
    runtime
        .get(key)
        .cloned()
        .or_else(|| std::env::var(key).ok())
}

fn normalize_meeting_stt_model(configured: Option<&str>) -> String {
    let configured = configured.map(str::trim).unwrap_or("");
    if configured.is_empty() || is_disabled_selector(configured) {
        return DEFAULT_MEETING_STT_MODEL.to_string();
    }
    let selected = engine::auxiliary_model_selection(engine::AuxiliaryRole::Stt, Some(configured));
    if selected.request_model == DEFAULT_MEETING_STT_MODEL {
        selected.request_model.to_string()
    } else {
        DEFAULT_MEETING_STT_MODEL.to_string()
    }
}

/// Persistent state for one meeting session, written to disk as JSON.
#[derive(Debug, Clone)]
pub(crate) struct MeetingSession {
    pub session_id: String,
    pub provider: String,
    pub meeting_url: String,
    pub bot_name: String,
    pub status: MeetingSessionStatus,
    pub started_at: String,
    pub ended_at: Option<String>,
    pub end_reason: Option<String>,
    pub transcript_chunks: Vec<String>,
    pub transcript_segments: Vec<TranscriptSegment>,
    pub speaker_signals: Vec<SpeakerSignal>,
    pub chat_messages: Vec<ChatMessage>,
    pub outbound_chat_texts: Vec<String>,
    pub pid: Option<u32>,
    pub stdin_pipe: Option<String>,
    /// Paths of audio chunk files whose STT failed (engine offline or error).
    /// Retried at finalize time if the engine becomes reachable.
    pub pending_audio_chunks: Vec<String>,
    /// Whether the STT engine was reachable when the session started.
    pub engine_was_reachable_at_start: bool,
    /// Whether local STT was proven suitable for live meeting transcripts.
    pub live_transcription_ready_at_start: bool,
    pub live_transcription_status_at_start: Option<Value>,
}

#[derive(Debug, Clone)]
pub(crate) struct ChatMessage {
    pub sender: String,
    pub text: String,
    pub timestamp: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct TranscriptSegment {
    pub timestamp: String,
    pub speaker_display: String,
    pub speaker_id: Option<String>,
    pub source: String,
    pub confidence: f32,
    pub text: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct SpeakerSignal {
    pub timestamp: String,
    pub speaker_display: String,
    pub speaker_id: Option<String>,
    pub source: String,
    pub confidence: f32,
}

impl TranscriptSegment {
    fn from_stt_text(text: String, speaker: Option<&SpeakerSignal>) -> Self {
        let (speaker_display, speaker_id, source, confidence) = speaker
            .map(|signal| {
                (
                    signal.speaker_display.clone(),
                    signal.speaker_id.clone(),
                    "stt_with_active_speaker".to_string(),
                    signal.confidence.min(0.65),
                )
            })
            .unwrap_or_else(|| ("unknown".to_string(), None, "stt".to_string(), 0.25));
        Self {
            timestamp: now_iso_string(),
            speaker_display,
            speaker_id,
            source,
            confidence,
            text,
        }
    }

    fn from_platform_event(event: &Value) -> Option<Self> {
        let text = event.get("text").and_then(Value::as_str)?.trim();
        if text.is_empty() {
            return None;
        }
        let speaker_display = sanitize_speaker_display(
            event
                .get("speaker")
                .or_else(|| event.get("speaker_display"))
                .and_then(Value::as_str)
                .unwrap_or("unknown"),
        );
        Some(Self {
            timestamp: event
                .get("ts")
                .or_else(|| event.get("timestamp"))
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(now_iso_string),
            speaker_display,
            speaker_id: event
                .get("speaker_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned),
            source: event
                .get("source")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or("platform_caption")
                .to_string(),
            confidence: event
                .get("confidence")
                .and_then(Value::as_f64)
                .map(|value| value.clamp(0.0, 1.0) as f32)
                .unwrap_or(0.85),
            text: text.to_string(),
        })
    }

    fn render_line(&self) -> String {
        format!(
            "[{}] {}: {} [source={} confidence={:.2}]",
            self.timestamp, self.speaker_display, self.text, self.source, self.confidence
        )
    }
}

impl SpeakerSignal {
    fn from_event(event: &Value) -> Option<Self> {
        let speaker_display = sanitize_speaker_display(
            event
                .get("speaker")
                .or_else(|| event.get("speaker_display"))
                .and_then(Value::as_str)?,
        );
        if speaker_display.eq_ignore_ascii_case("unknown") {
            return None;
        }
        Some(Self {
            timestamp: event
                .get("ts")
                .or_else(|| event.get("timestamp"))
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned)
                .unwrap_or_else(now_iso_string),
            speaker_display,
            speaker_id: event
                .get("speaker_id")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .map(ToOwned::to_owned),
            source: event
                .get("source")
                .and_then(Value::as_str)
                .filter(|value| !value.trim().is_empty())
                .unwrap_or("platform_active_speaker")
                .to_string(),
            confidence: event
                .get("confidence")
                .and_then(Value::as_f64)
                .map(|value| value.clamp(0.0, 1.0) as f32)
                .unwrap_or(0.55),
        })
    }
}

fn sanitize_speaker_display(value: &str) -> String {
    let cleaned = value
        .replace('\n', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let cleaned = cleaned
        .trim_matches(|ch: char| ch == ':' || ch == '-' || ch == '|' || ch.is_whitespace())
        .trim();
    if cleaned.is_empty() {
        "unknown".to_string()
    } else {
        cleaned.chars().take(96).collect()
    }
}

impl MeetingSession {
    pub(crate) fn new(config: &MeetingSessionConfig) -> Self {
        let session_id = format!(
            "meeting-{}-{}",
            config.provider.as_str(),
            now_epoch_millis()
        );
        Self {
            session_id,
            provider: config.provider.as_str().to_string(),
            meeting_url: config.meeting_url.clone(),
            bot_name: config.bot_name.clone(),
            status: MeetingSessionStatus::Joining,
            started_at: now_iso_string(),
            ended_at: None,
            end_reason: None,
            transcript_chunks: Vec::new(),
            transcript_segments: Vec::new(),
            speaker_signals: Vec::new(),
            chat_messages: Vec::new(),
            outbound_chat_texts: Vec::new(),
            pid: None,
            stdin_pipe: None,
            pending_audio_chunks: Vec::new(),
            engine_was_reachable_at_start: false,
            live_transcription_ready_at_start: false,
            live_transcription_status_at_start: None,
        }
    }

    pub(crate) fn to_json(&self) -> Value {
        json!({
            "session_id": self.session_id,
            "provider": self.provider,
            "meeting_url": self.meeting_url,
            "bot_name": self.bot_name,
            "status": self.status.as_str(),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "end_reason": self.end_reason,
            "transcript_chunk_count": self.transcript_chunks.len(),
            "transcript_segment_count": self.transcript_segments.len(),
            "speaker_signal_count": self.speaker_signals.len(),
            "chat_message_count": self.chat_messages.len(),
            "transcript_chunks": self.transcript_chunks,
            "transcript_segments": self.transcript_segments,
            "speaker_signals": self.speaker_signals,
            "chat_messages": self.chat_messages.iter().map(|m| json!({
                "sender": m.sender,
                "text": m.text,
                "timestamp": m.timestamp,
            })).collect::<Vec<_>>(),
            "outbound_chat_texts": &self.outbound_chat_texts,
            "pid": self.pid,
            "stdin_pipe": self.stdin_pipe,
            "pending_audio_chunks": self.pending_audio_chunks,
            "engine_was_reachable_at_start": self.engine_was_reachable_at_start,
            "live_transcription_ready_at_start": self.live_transcription_ready_at_start,
            "live_transcription_status_at_start": self.live_transcription_status_at_start,
        })
    }

    pub(crate) fn save(&self, root: &Path) -> Result<()> {
        let dir = meeting_sessions_dir(root);
        fs::create_dir_all(&dir)?;
        let path = dir.join(format!("{}.json", self.session_id));
        fs::write(&path, serde_json::to_string_pretty(&self.to_json())?)?;
        Ok(())
    }

    /// Build the full transcript from all chunks.
    pub(crate) fn full_transcript(&self) -> String {
        if !self.transcript_segments.is_empty() {
            return self
                .transcript_segments
                .iter()
                .map(TranscriptSegment::render_line)
                .collect::<Vec<_>>()
                .join("\n");
        }
        self.transcript_chunks.join("\n")
    }

    pub(crate) fn push_stt_transcript(&mut self, text: String, speaker: Option<&SpeakerSignal>) {
        if text.trim().is_empty() {
            return;
        }
        self.transcript_chunks.push(text.clone());
        self.transcript_segments
            .push(TranscriptSegment::from_stt_text(text, speaker));
    }

    pub(crate) fn push_platform_transcript(&mut self, segment: TranscriptSegment) {
        if segment.text.trim().is_empty() {
            return;
        }
        self.transcript_chunks.push(segment.text.clone());
        self.transcript_segments.push(segment);
    }

    /// Build the full chat log.
    pub(crate) fn full_chat_log(&self) -> String {
        self.chat_messages
            .iter()
            .map(|msg| format!("[{}] {}: {}", msg.timestamp, msg.sender, msg.text))
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Check if a chat message mentions the meeting bot.
    /// Returns true if a known bot mention appears with a word boundary on both sides
    /// (so "@ctoxbar" doesn't match, but "@INF Yoda Notetaker" or "@ctox!" do).
    pub(crate) fn is_mention(text: &str) -> bool {
        let lower = normalize_chat_text(text).to_lowercase();
        for prefix in ["inf yoda notetaker", "inf yoda", "ctox"] {
            if let Some(rest) = lower.strip_prefix(prefix) {
                let rest = rest.trim_start();
                if rest.starts_with(':') || rest.starts_with('-') || rest.starts_with(',') {
                    return true;
                }
            }
        }
        for needle in ["@ctox", "@inf yoda", "@inf yoda notetaker"] {
            let mut search_from = 0;
            while let Some(pos) = lower[search_from..].find(needle) {
                let abs_pos = search_from + pos;
                let after = abs_pos + needle.len();
                // Word boundary check: char after must be non-alphanumeric (or end of string)
                let bounded = lower[after..]
                    .chars()
                    .next()
                    .map(|c| !c.is_ascii_alphanumeric())
                    .unwrap_or(true);
                if bounded {
                    return true;
                }
                search_from = after;
            }
        }
        false
    }

    /// Check if a chat message likely originated from this bot itself.
    /// Used to prevent self-loop when the bot's own replies appear in the chat.
    pub(crate) fn is_own_message(&self, sender: &str, text: &str) -> bool {
        is_own_message_text(&self.bot_name, &self.outbound_chat_texts, sender, text)
    }
}

fn is_own_message_text(
    bot_name: &str,
    outbound_chat_texts: &[String],
    sender: &str,
    text: &str,
) -> bool {
    let bot_name_lower = normalize_chat_text(bot_name).to_lowercase();
    let bot_name_lower = bot_name_lower.trim();
    if bot_name_lower.is_empty() {
        return false;
    }
    let sender_lower = normalize_chat_text(sender).to_lowercase();
    // Match if sender contains the bot name (sender field may include
    // role suffixes like "(Host)" or be wrapped in other text)
    if sender_lower.contains(bot_name_lower) {
        return true;
    }
    if matches!(
        sender_lower.trim(),
        "you" | "me" | "ich" | "du" | "ctox" | "ctox notetaker"
    ) {
        return true;
    }
    // Some chat scrapers misattribute and put the sender in the text;
    // match if text starts with the bot name + colon/dash separator
    let text_lower = normalize_chat_text(text).to_lowercase();
    if outbound_chat_texts.iter().any(|sent| {
        let sent = normalize_chat_text(sent).trim().to_lowercase();
        !sent.is_empty() && text_lower.contains(&sent)
    }) {
        return true;
    }
    false
}

fn normalize_chat_text(value: &str) -> String {
    value
        .replace('\u{00a0}', " ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// Playwright meeting runner script generation
// ---------------------------------------------------------------------------
//
// The templates below are transplanted from the ScreenApp meeting-bot reference
// implementation (~/Downloads/meeting-bot).  Key architectural decisions kept:
//
//   * Google Meet + Zoom → getDisplayMedia + MediaRecorder (in-browser capture)
//   * Microsoft Teams    → ffmpeg + X11grab + PulseAudio (out-of-process capture)
//   * Participant detection per provider uses the exact DOM queries from the
//     reference (data-avatar-count, badge-div .egzc7c, #wc-footer, etc.)
//   * Silence detection via AudioContext+Analyser (Google/Zoom) or parec (Teams)
//   * Each provider's join-flow includes retry logic, device-notification
//     dismissal, and lobby-mode detection with the same text constants.
//
// Placeholders: __MEETING_URL__, __BOT_NAME__, __PROVIDER__, __CHUNK_SECONDS__,
// __MAX_DURATION_MS__, __JOIN_SCRIPT__, __CHAT_SCRAPE_SCRIPT__,
// __SEND_CHAT_SCRIPT__, __RECORDING_SCRIPT__

/// The runner template uses placeholder tokens (__MEETING_URL__ etc.) instead of
/// Rust format placeholders to avoid brace-escaping conflicts with JavaScript.
// The Playwright meeting runner lives as a real JS file so it can be
// linted and reviewed as JavaScript; placeholders (__MEETING_URL__ etc.)
// are substituted at spawn time exactly as before.
const MEETING_RUNNER_TEMPLATE: &str = include_str!("runner/meeting_runner.mjs");

/// Build a long-running Node.js Playwright script that:
/// 1. Joins the meeting as a guest
/// 2. Captures audio via getDisplayMedia + MediaRecorder
/// 3. Polls the meeting chat for new messages
/// 4. Emits JSON-lines events on stdout
/// 5. Accepts JSON commands on stdin (e.g., send_chat)
pub(crate) fn build_meeting_runner_script(config: &MeetingSessionConfig) -> Result<String> {
    let url = serde_json::to_string(&config.meeting_url)?;
    let bot_name = serde_json::to_string(&config.bot_name)?;
    let provider = config.provider.as_str();
    let chunk_seconds = config.audio_chunk_seconds;
    let max_duration_ms = config.max_duration_minutes * 60 * 1000;

    let join_script = match config.provider {
        MeetingProvider::GoogleMeet => build_google_meet_join_script(),
        MeetingProvider::MicrosoftTeams => build_teams_join_script(),
        MeetingProvider::Zoom => build_zoom_join_script(),
    };

    let chat_scrape_script = match config.provider {
        MeetingProvider::GoogleMeet => build_google_meet_chat_scraper(),
        MeetingProvider::MicrosoftTeams => build_teams_chat_scraper(),
        MeetingProvider::Zoom => build_zoom_chat_scraper(),
    };

    let send_chat_script = match config.provider {
        MeetingProvider::GoogleMeet => build_google_meet_chat_sender(),
        MeetingProvider::MicrosoftTeams => build_teams_chat_sender(),
        MeetingProvider::Zoom => build_zoom_chat_sender(),
    };

    // Use string replacement instead of format! to avoid brace escaping issues
    // with JavaScript code that heavily uses { and }.
    Ok(MEETING_RUNNER_TEMPLATE
        .replace("__MEETING_URL__", &url)
        .replace("__BOT_NAME__", &bot_name)
        .replace("__PROVIDER__", provider)
        .replace("__CHUNK_SECONDS__", &chunk_seconds.to_string())
        .replace("__MAX_DURATION_MS__", &max_duration_ms.to_string())
        .replace("__JOIN_SCRIPT__", join_script)
        .replace("__CHAT_SCRAPE_SCRIPT__", chat_scrape_script)
        .replace("__SEND_CHAT_SCRIPT__", send_chat_script))
}

// ---------------------------------------------------------------------------
// Provider-specific join scripts (injected into the Playwright runner)
// ---------------------------------------------------------------------------

fn build_google_meet_join_script() -> &'static str {
    r#"
// Google Meet join flow — transplanted from ScreenApp meeting-bot reference
try {
  const detectPage = async () => {
    const currentUrl = page.url();
    if (currentUrl.startsWith("https://accounts.google.com/")) {
      return "SIGN_IN_PAGE";
    }
    if (currentUrl.includes("workspace.google.com/products/meet")) {
      return "UNSUPPORTED_PAGE";
    }
    if (!currentUrl.includes("meet.google.com")) {
      return "UNSUPPORTED_PAGE";
    }
    return "GOOGLE_MEET_PAGE";
  };

  const initialPageStatus = await detectPage();
  if (initialPageStatus === "SIGN_IN_PAGE") {
    throw new Error("Meeting requires sign in");
  }
  if (initialPageStatus === "UNSUPPORTED_PAGE") {
    throw new Error("Google Meet redirected to unsupported page: " + page.url());
  }

  // 1. Dismiss "Continue without microphone and camera" (with retry)
  try {
    const retryClick = async (desc, fn, retries = 1, wait = 15000) => {
      for (let i = 0; i <= retries; i++) {
        try { await fn(); return; } catch (e) {
          if (i === retries) throw e;
          await page.waitForTimeout(wait);
        }
      }
    };
    await retryClick(
      "Continue without microphone and camera",
      async () => {
        const button = page.getByRole("button", {
          name: /Continue without microphone and camera|Ohne Mikrofon und Kamera fortfahren|Mikrofon und Kamera nicht verwenden/i
        }).first();
        await button.waitFor({ timeout: 30000 });
        await button.click();
      }
    );
  } catch { /* may not appear */ }

  // 2. Verify we are on a Google Meet page (not redirected to sign-in)
  const pageStatus = await detectPage();
  if (pageStatus === "SIGN_IN_PAGE") {
    throw new Error("Meeting requires sign in");
  }
  if (pageStatus === "UNSUPPORTED_PAGE") {
    throw new Error("Google Meet redirected to unsupported page: " + page.url());
  }

  // 3. Wait for name input and fill it (with retry)
  const nameInputSelectors = [
    'input[type="text"][aria-label="Your name"]',
    'input[type="text"][aria-label*="name" i]',
    'input[type="text"][aria-label*="Name" i]',
    'input[type="text"][placeholder*="name" i]',
    'input[type="text"][placeholder*="Name" i]',
    'input[type="text"]',
  ];
  let filledName = false;
  try {
    const retryWait = async (desc, fn, retries = 3, wait = 15000, onError) => {
      for (let i = 0; i <= retries; i++) {
        try { await fn(); return; } catch (e) {
          if (onError) try { await onError(); } catch {}
          if (i === retries) throw e;
          await page.waitForTimeout(wait);
        }
      }
    };
    await retryWait(
      "Name input field",
      async () => {
        for (const selector of nameInputSelectors) {
          const input = page.locator(selector).first();
          if (await input.isVisible({ timeout: 1000 }).catch(() => false)) return;
        }
        throw new Error("name input not visible");
      },
      3,
      15000
    );
  } catch (err) {
    emit({ type: "warning", message: "Name input not found: " + err.message });
  }

  for (const selector of nameInputSelectors) {
    try {
      const input = page.locator(selector).first();
      if (await input.isVisible({ timeout: 1000 }).catch(() => false)) {
        await input.fill(botName);
        filledName = true;
        break;
      }
    } catch {}
  }
  if (filledName) {
    await page.waitForTimeout(2000);
  }

  // 4. Click join button (Ask to join / Join now / Join anyway) — with retry
  {
    const possibleTexts = [
      "Ask to join",
      "Join now",
      "Join anyway",
      "Teilnahme anfragen",
      "Jetzt teilnehmen",
      "Teilnehmen",
      "Trotzdem teilnehmen",
    ];
    let buttonClicked = false;
    for (let attempt = 0; attempt <= 3 && !buttonClicked; attempt++) {
      for (const text of possibleTexts) {
        try {
          const btn = page.locator("button", { hasText: new RegExp(text, "i") }).first();
          if (await btn.isVisible({ timeout: 3000 }).catch(() => false)) {
            await btn.click({ timeout: 5000 });
            buttonClicked = true;
            break;
          }
        } catch { /* try next text */ }
      }
      if (!buttonClicked) await page.waitForTimeout(15000);
    }
    if (!buttonClicked) {
      emit({ type: "warning", message: "Could not find join button" });
    }
  }

  // 5. Wait at lobby — detect admission via People button + participant count
  //    Transplanted from reference: 6-method participant detection
  {
    const LOBBY_HOST_TEXT = "Please wait until a meeting host brings you";
    const REQUEST_DENIED = "Someone in the call denied your request to join";
    const REQUEST_TIMEOUT = "No one responded to your request to join the call";
    const wanderingTime = Math.min(10 * 60 * 1000, maxDurationMs);
    const lobbyResult = await new Promise((resolve) => {
      const timeout = setTimeout(() => { clearInterval(interval); resolve(false); }, wanderingTime);
      const interval = setInterval(async () => {
        try {
          // Check for denied/timeout
          const bodyText = await page.evaluate(() => document.body.innerText);
          if (bodyText.includes(REQUEST_DENIED)) {
            clearInterval(interval); clearTimeout(timeout); resolve(false); return;
          }
          if (bodyText.includes(REQUEST_TIMEOUT)) {
            clearInterval(interval); clearTimeout(timeout); resolve(false); return;
          }
          if (
            bodyText.includes(LOBBY_HOST_TEXT)
            || bodyText.includes("Jemand wird dich")
            || bodyText.includes("Jemand wird Sie")
            || bodyText.includes("Teilnahme anfragen")
            || bodyText.includes("Bitte warten")
          ) return; // still waiting

          // Check for People button or Leave call button
          const detected = await page.evaluate(() => {
            try {
              const peopleBtn = document.querySelector('button[aria-label^="People"]')
                || document.querySelector('button[aria-label*="People"]')
                || document.querySelector('button[aria-label*="Teilnehmer"]');
              const leaveBtn = document.querySelector('button[aria-label="Leave call"]')
                || document.querySelector('button[aria-label*="Verlassen"]')
                || document.querySelector('button[aria-label*="Anruf verlassen"]');

              if (!peopleBtn && !leaveBtn) return false;

              // Check participant count via data-avatar-count
              if (peopleBtn) {
                const roots = [peopleBtn, peopleBtn.parentElement, peopleBtn.parentElement?.parentElement].filter(Boolean);
                for (const root of roots) {
                  const avatar = root.querySelector("[data-avatar-count]");
                  if (avatar) {
                    const count = Number(avatar.getAttribute("data-avatar-count"));
                    if (!isNaN(count) && count >= 1) return true;
                  }
                  // Fallback: badge div with class egzc7c
                  const badge = root.querySelector("div.egzc7c");
                  if (badge) {
                    const text = (badge.innerText || badge.textContent || "").trim();
                    if (/^\d+$/.test(text) && Number(text) >= 1) return true;
                  }
                }
              }

              // Fallback: Leave call button present + no lobby text
              if (leaveBtn) {
                const bt = document.body.innerText || "";
                if (!bt.includes("Asking to join") && !bt.includes("You're the only one here")) {
                  return true;
                }
              }
              return false;
            } catch { return false; }
          });

          if (detected) {
            clearInterval(interval); clearTimeout(timeout); resolve(true);
          }
        } catch { /* retry next tick */ }
      }, 20000);
    });

    if (!lobbyResult) {
      const bodyText = await page.evaluate(() => document.body.innerText);
      emit({ type: "error", message: "Lobby admission failed", bodyText: (bodyText || "").substring(0, 500) });
    }
  }

  // 6. Dismiss "Got it" modals (loop until all gone)
  try {
    await page.waitForSelector('button:has-text("Got it")', { timeout: 15000 });
    let consecutiveNoChange = 0;
    let prevCount = -1;
    while (true) {
      const btns = await page.locator('button:visible', { hasText: "Got it" }).all();
      if (btns.length === 0) break;
      if (btns.length === prevCount) { consecutiveNoChange++; if (consecutiveNoChange >= 2) break; }
      else consecutiveNoChange = 0;
      prevCount = btns.length;
      for (const btn of btns) { try { await btn.click({ timeout: 5000 }); await page.waitForTimeout(2000); } catch {} }
      await page.waitForTimeout(2000);
    }
  } catch { /* modals may be missing */ }

  // 7. Dismiss device notifications (Microphone/Camera not found)
  try {
    const hasNotif = await page.evaluate(() =>
      document.body.innerText.includes("Microphone not found") ||
      document.body.innerText.includes("Camera not found") ||
      document.body.innerText.includes("Make sure your microphone is plugged in")
    );
    if (hasNotif) {
      await page.evaluate(() => {
        const allButtons = Array.from(document.querySelectorAll("button"));
        allButtons.filter(btn => {
          const label = btn.getAttribute("aria-label");
          const hasIcon = btn.querySelector("svg") !== null;
          return (label?.toLowerCase().includes("close") ||
                  label?.toLowerCase().includes("dismiss") ||
                  (hasIcon && btn.offsetParent !== null && btn.innerText === ""));
        }).forEach(btn => { if (btn.offsetParent !== null) btn.click(); });
      });
    }
  } catch {}
} catch (err) {
  emit({ type: "error", message: "Google Meet join error: " + err.message });
}
"#
}

fn build_teams_join_script() -> &'static str {
    r#"
// Microsoft Teams join flow — transplanted from ScreenApp meeting-bot reference
// Note: Teams uses ffmpeg+PulseAudio for recording, not getDisplayMedia.
// The browser is launched with --use-fake-ui-for-media-stream, --kiosk.
try {
  const teamsScopes = () => [page, ...page.frames().filter((frame) => frame !== page.mainFrame())];
  const warmUpTeamsMediaDevices = async () => {
    try {
      await page.evaluate(async () => {
        if (!navigator.mediaDevices?.getUserMedia) return false;
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: true });
        stream.getTracks().forEach((track) => track.stop());
        return true;
      });
    } catch {}
  };
  const waitForTeamsPreJoinReadiness = async (timeoutMs = 45000) => {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
      for (const scope of teamsScopes()) {
        const ready = await scope.evaluate(() => {
          const visible = (el) => {
            try {
              const rect = el.getBoundingClientRect();
              const style = window.getComputedStyle(el);
              return rect.width > 0 && rect.height > 0 && style.visibility !== "hidden" && style.display !== "none";
            } catch { return false; }
          };
          const hasName = Array.from(document.querySelectorAll("input, textarea, [contenteditable='true'], [role='textbox']"))
            .some((el) => visible(el) && (el.value || el.textContent || el.getAttribute("aria-label") || "").length >= 0);
          const hasJoin = Array.from(document.querySelectorAll("button"))
            .some((btn) => visible(btn) && /join|teilnehmen|beitreten|ask to join/i.test(btn.innerText || btn.textContent || btn.getAttribute("aria-label") || ""));
          return hasName || hasJoin;
        }).catch(() => false);
        if (ready) return true;
      }
      await page.waitForTimeout(1000);
    }
    return false;
  };

  await warmUpTeamsMediaDevices();

  // 1. Click "Join from browser" / "Continue on this browser"
  const joinButtonSelectors = [
    'button[aria-label="Join meeting from this browser"]',
    'button[aria-label="Continue on this browser"]',
    'button[aria-label="Join on this browser"]',
    'button:has-text("Continue on this browser")',
    'button:has-text("Join from browser")',
    'button:has-text("In diesem Browser fortfahren")',
    'button:has-text("In diesem Browser teilnehmen")',
  ];
  let browserBtnClicked = false;
  const visibleTextInputInScope = async (scope) => {
    try {
      return await scope.evaluate(() => {
        return Array.from(document.querySelectorAll("input")).some((el) => {
          const rect = el.getBoundingClientRect();
          const style = window.getComputedStyle(el);
          return rect.width > 80 && rect.height > 20 && style.visibility !== "hidden" && style.display !== "none";
        });
      });
    } catch { return false; }
  };
  let alreadyOnPrejoin = false;
  for (const scope of teamsScopes()) {
    if (await visibleTextInputInScope(scope)) { alreadyOnPrejoin = true; break; }
  }
  if (!alreadyOnPrejoin) {
    for (const sel of joinButtonSelectors) {
      try {
        const button = page.locator(sel).first();
        if (await button.isVisible({ timeout: 3000 }).catch(() => false)) {
          await button.click({ force: true });
          browserBtnClicked = true;
          break;
        }
      } catch { continue; }
    }
  }
  if (!browserBtnClicked && !alreadyOnPrejoin) {
    emit({ type: "warning", message: "Join from browser button not found, proceeding" });
  }
  await waitForTeamsPreJoinReadiness();

  // 2. Fill name input (Teams light meetings/localized variants)
  try {
    const nameInputSelectors = [
      'input[data-tid="prejoin-display-name-input"]',
      'input[placeholder*="name" i]',
      'input[placeholder*="Namen" i]',
      'input[type="text"]',
    ];
    let filledName = false;
    for (const scope of teamsScopes()) {
      for (const sel of nameInputSelectors) {
        const nameInput = scope.locator(sel).first();
        if (await nameInput.isVisible({ timeout: 2000 }).catch(() => false)) {
          await nameInput.fill(botName);
          filledName = true;
          break;
        }
      }
      if (filledName) break;
    }
    if (!filledName) {
      for (const scope of teamsScopes()) {
        const filled = await scope.evaluate((name) => {
          const candidates = Array.from(document.querySelectorAll("input, textarea, [contenteditable='true'], [role='textbox']"));
          for (const el of candidates) {
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            if (rect.width <= 80 || rect.height <= 20 || style.visibility === "hidden" || style.display === "none") continue;
            el.focus();
            if ("value" in el) el.value = name;
            else el.textContent = name;
            el.dispatchEvent(new Event("input", { bubbles: true }));
            el.dispatchEvent(new Event("change", { bubbles: true }));
            return true;
          }
          return false;
        }, botName).catch(() => false);
        if (filled) { filledName = true; break; }
      }
    }
    if (!filledName) {
      // Teams light-meetings sometimes renders the guest name control through
      // a localized/translated surface that Playwright cannot see as a normal
      // input. The prejoin layout is stable enough for this fallback.
      await page.mouse.click(640, 200);
      await page.keyboard.press(process.platform === "darwin" ? "Meta+A" : "Control+A");
      await page.keyboard.type(botName);
      filledName = true;
    }
    if (!filledName) throw new Error("no visible Teams display-name input");
    await page.waitForTimeout(1000);
  } catch (err) {
    emit({ type: "warning", message: "Teams name input not found: " + err.message });
  }

  // 2b. Select computer audio so the "Join now" button becomes enabled.
  try {
    const audioLabels = [
      /Computer audio/i,
      /Computeraudio/i,
      /Use computer audio/i,
    ];
    let selectedAudio = false;
    for (const scope of teamsScopes()) {
      for (const label of audioLabels) {
        const option = scope.getByText(label).first();
        if (await option.isVisible({ timeout: 2000 }).catch(() => false)) {
          await option.click({ force: true });
          selectedAudio = true;
          break;
        }
      }
      if (selectedAudio) break;
    }
    if (!selectedAudio) {
      for (const scope of teamsScopes()) {
        const clicked = await scope.evaluate(() => {
          const cards = Array.from(document.querySelectorAll("button, label, div"));
          for (const el of cards) {
            const text = (el.innerText || el.textContent || "").trim();
            if (!/Computer audio|Computeraudio|Use computer audio/i.test(text)) continue;
            const rect = el.getBoundingClientRect();
            const style = window.getComputedStyle(el);
            if (rect.width <= 50 || rect.height <= 20 || style.visibility === "hidden" || style.display === "none") continue;
            el.click();
            return true;
          }
          return false;
        }).catch(() => false);
        if (clicked) {
          selectedAudio = true;
          break;
        }
      }
    }
    if (selectedAudio) await page.waitForTimeout(1000);
  } catch (err) {
    emit({ type: "warning", message: "Teams audio option not selected: " + err.message });
  }

  // 3. Keep the injected transcript camera on, mute microphone
  try {
    await page.waitForTimeout(2000);
    // Microphone mute
    const micSelectors = [
      'input[data-tid="toggle-mute"]:not([checked])',
      'input[type="checkbox"][title*="Mute mic" i]',
      'input[role="switch"][data-tid="toggle-mute"]',
      'button[aria-label*="Mute microphone" i]',
      'button[aria-label*="Mute mic" i]',
    ];
    for (const sel of micSelectors) {
      const el = page.locator(sel).first();
      if (await el.isVisible({ timeout: 2000 }).catch(() => false)) {
        await el.click(); await page.waitForTimeout(500); break;
      }
    }
  } catch { /* device toggles best-effort */ }

  // 4. Click Join button (with retry)
  {
    const possibleTexts = ["Join now", "Join", "Ask to join", "Join meeting", "Jetzt teilnehmen", "Teilnehmen"];
    let joinClicked = false;
    for (let attempt = 0; attempt <= 3 && !joinClicked; attempt++) {
      for (const scope of teamsScopes()) {
        for (const text of possibleTexts) {
          try {
            const btn = scope.getByRole("button", { name: new RegExp(text, "i") });
            if (await btn.isVisible({ timeout: 2000 }).catch(() => false)) {
              await btn.click(); joinClicked = true; break;
            }
          } catch {}
        }
        if (joinClicked) break;
      }
      if (!joinClicked) {
        for (const scope of teamsScopes()) {
          const clicked = await scope.evaluate((labels) => {
            const buttons = Array.from(document.querySelectorAll("button"));
            for (const button of buttons) {
              const text = (button.innerText || button.getAttribute("aria-label") || "").trim();
              if (!labels.some((label) => text.toLowerCase().includes(label.toLowerCase()))) continue;
              button.click();
              return true;
            }
            return false;
          }, possibleTexts).catch(() => false);
          if (clicked) {
            joinClicked = true;
            break;
          }
        }
      }
      if (!joinClicked && attempt === 1) {
        await page.mouse.click(1060, 600);
        joinClicked = true;
      }
      if (!joinClicked) await page.waitForTimeout(15000);
    }
    if (!joinClicked) emit({ type: "warning", message: "Could not find Teams join button" });
    await page.keyboard.press(process.platform === "darwin" ? "Meta+Shift+M" : "Control+Shift+M").catch(() => {});
  }

  // 5. Wait for lobby admission (Leave button appears)
  {
    const DENIED_TEXT = "Sorry, but you were denied access to the meeting";
    const wanderingTime = Math.min(10 * 60 * 1000, maxDurationMs);
    try {
      const leaveBtn = page.getByRole("button", { name: /Leave|Verlassen/i });
      await leaveBtn.waitFor({ timeout: wanderingTime });
    } catch {
      const bodyText = await page.evaluate(() => document.body.innerText);
      const denied = (bodyText || "").includes(DENIED_TEXT);
      emit({ type: "error", message: "Teams lobby failed", denied, bodyText: (bodyText || "").substring(0, 500) });
    }
  }

  // 6. Dismiss Close buttons (notifications/device checks) — with loop
  try {
    await page.waitForSelector('button[aria-label=Close]', { timeout: 5000 });
    await page.click('button[aria-label=Close]', { timeout: 2000 });
  } catch {}
  try {
    let prevCount = -1;
    let noChange = 0;
    while (true) {
      const btns = await page.locator('button[title="Close"]:visible').all();
      if (btns.length === 0) break;
      if (btns.length === prevCount) { noChange++; if (noChange >= 2) break; }
      else noChange = 0;
      prevCount = btns.length;
      for (const btn of btns) { try { await btn.click({ timeout: 5000 }); await page.waitForTimeout(2000); } catch {} }
      await page.waitForTimeout(2000);
    }
  } catch {}

  // 7. Wait for audio to stabilize before recording
  await page.waitForTimeout(5000);
} catch (err) {
  emit({ type: "error", message: "Teams join error: " + err.message });
}
"#
}

fn build_zoom_join_script() -> &'static str {
    r##"
// Zoom join flow — direct web client flow with Vexa-style readiness checks.
try {
  // Block .exe downloads
  await page.route("**/*.exe", (route) => {
    emit({ type: "status", status: "blocked_exe_download", url: route.request().url() });
  });

  // 1. Accept cookies
  try {
    await page.waitForTimeout(3000);
    const acceptCookies = page.locator("button", { hasText: /Accept Cookies|Cookies akzeptieren|Alle Cookies akzeptieren/i }).first();
    await acceptCookies.waitFor({ timeout: 5000 });
    await acceptCookies.click({ force: true });
  } catch { /* may not appear */ }

  if (!page.url().includes("/wc/")) {
    await page.goto(buildZoomWebClientUrl(meetingUrl), { waitUntil: "domcontentloaded", timeout: 60000 });
  }
  await page.waitForTimeout(5000);

  const text = (await page.evaluate(() => document.body?.innerText || "").catch(() => "")).toLowerCase();
  if (/sign in to join|only authenticated users|not authorized|meeting authentication/i.test(text)) {
    emit({ type: "join_failed", provider, reason: "zoom_requires_authentication", bodyText: text.substring(0, 500) });
  }

  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const allow = page.getByRole("button", { name: /Allow/i }).first();
      if (await allow.isVisible({ timeout: 1500 }).catch(() => false)) await allow.click({ force: true });
    } catch {}
  }

  const zoomScopes = () => [page, ...page.frames().filter((frame) => frame !== page.mainFrame())];
  const findVisibleLocator = async (selectors, timeout = 1500) => {
    for (const scope of zoomScopes()) {
      for (const selector of selectors) {
        try {
          const locator = scope.locator(selector).first();
          if (await locator.isVisible({ timeout }).catch(() => false)) return locator;
        } catch {}
      }
    }
    return null;
  };

  const nameInput = await findVisibleLocator([
    "#input-for-name",
    'input[aria-label*="name" i]',
    'input[placeholder*="name" i]',
    'input[type="text"]',
  ], 30000);
  if (nameInput) {
    await nameInput.click({ force: true });
    await nameInput.fill("").catch(() => {});
    await page.keyboard.type(botName, { delay: 30 });
  } else {
    emit({ type: "warning", message: "Zoom name input not found" });
  }

  const passcodeInput = await findVisibleLocator([
    "#input-for-pwd",
    'input[type="password"]',
    'input[placeholder*="passcode" i]',
    'input[aria-label*="passcode" i]',
  ], 1000);
  if (passcodeInput && !new URL(buildZoomWebClientUrl(meetingUrl)).searchParams.get("pwd")) {
    emit({ type: "warning", message: "Zoom passcode field visible but no passcode was present in the meeting URL" });
  }

  const joinSelectors = [
    "button.preview-join-button",
    'button[type="submit"]',
    'button:has-text("Join")',
    'button:has-text("Beitreten")',
  ];
  let joinedClicked = false;
  for (let attempt = 0; attempt < 5 && !joinedClicked; attempt++) {
    const joinButton = await findVisibleLocator(joinSelectors, 2000);
    if (joinButton) {
      try {
        await joinButton.waitFor({ state: "visible", timeout: 5000 });
        const enabled = await joinButton.evaluate((btn) => !btn.disabled && !btn.classList.contains("disabled")).catch(() => true);
        if (enabled) {
          joinedClicked = await joinButton.evaluate((btn) => { btn.click(); return true; }).catch(() => false);
          if (!joinedClicked) {
            await joinButton.click({ force: true, timeout: 3000 });
            joinedClicked = true;
          }
        }
      } catch {}
    }
    if (!joinedClicked) await page.waitForTimeout(2000);
  }
  if (!joinedClicked) emit({ type: "error", message: "Zoom join button not found or disabled" });

  await page.waitForTimeout(5000);

  const previewStopVideo = await findVisibleLocator([
    'button[aria-label*="Stop Video" i]',
    'button[title*="Stop Video" i]',
  ], 1000);
  if (previewStopVideo) await previewStopVideo.click({ force: true }).catch(() => {});

  const wanderingTime = Math.min(10 * 60 * 1000, maxDurationMs);
  const deadline = Date.now() + wanderingTime;
  while (Date.now() < deadline) {
    const state = await page.evaluate(() => {
      const body = (document.body?.innerText || "").toLowerCase();
      const leave = document.querySelector('button[aria-label*="Leave" i], button[title*="Leave" i]');
      const footer = document.querySelector("#wc-footer");
      return {
        admitted: Boolean(leave) || /participants?|teilnehmer/i.test(footer?.textContent || ""),
        waiting: /please wait|waiting room|let you in soon|we've let them know|host has not joined/i.test(body),
        denied: /removed|denied|no one responded|meeting has ended/i.test(body),
        bodyText: body.substring(0, 500),
      };
    });
    if (state.admitted) break;
    if (state.denied) {
      emit({ type: "error", message: "Zoom lobby failed", bodyText: state.bodyText });
      break;
    }
    if (state.waiting) emit({ type: "status", status: "waiting_lobby", provider });
    await page.waitForTimeout(3000);
  }

  await dismissZoomPopups(page, 30000);
} catch (err) {
  emit({ type: "error", message: "Zoom join error: " + err.message });
}
"##
}

// ---------------------------------------------------------------------------
// Provider-specific chat scraping (runs inside page.evaluate)
// ---------------------------------------------------------------------------

fn build_google_meet_chat_scraper() -> &'static str {
    r#"
      const messages = [];
      // Google Meet chat messages
      const chatMsgs = document.querySelectorAll('[data-message-id]');
      for (const el of chatMsgs) {
        const senderEl = el.querySelector('[data-sender-name]');
        const textEl = el.querySelector('[data-message-text]');
        const sender = senderEl?.getAttribute('data-sender-name') || senderEl?.textContent?.trim() || 'Unknown';
        const text = textEl?.textContent?.trim() || el.textContent?.trim() || '';
        if (text) messages.push({ sender, text, ts: new Date().toISOString() });
      }
      // Fallback: try aria-label based selectors
      if (messages.length === 0) {
        const items = document.querySelectorAll('[data-is-chat-message="true"], [jsname] [role="listitem"]');
        for (const item of items) {
          const clone = item.cloneNode(true);
          clone.querySelectorAll('[aria-label*="Pinned" i], [aria-label*="pin" i], button, svg').forEach((node) => node.remove());
          const lines = (clone.innerText || clone.textContent || '').split(/\n+/).map((line) => line.trim()).filter(Boolean);
          const sender = lines.length > 1 && lines[0].length <= 80 ? lines[0] : 'Participant';
          const text = lines.length > 1 ? lines.slice(1).join(' ') : lines.join(' ');
          if (text) messages.push({ sender, text, ts: new Date().toISOString() });
        }
      }
      return messages;
    "#
}

fn build_teams_chat_scraper() -> &'static str {
    r#"
      const messages = [];
      const chatItems = document.querySelectorAll('[data-tid="chat-pane-message"], [role="listitem"]');
      for (const el of chatItems) {
        const senderEl = el.querySelector('[data-tid="message-author-name"]') || el.querySelector('.ui-chat__message__author') || el.querySelector('[class*="author" i], [class*="sender" i]');
        const textEl = el.querySelector('[data-tid="message-body"]') || el.querySelector('.ui-chat__message__content') || el.querySelector('[class*="message-body" i], [class*="content" i]');
        const sender = senderEl?.textContent?.trim() || 'Unknown';
        const text = textEl?.textContent?.trim() || el.textContent?.trim() || '';
        if (text) messages.push({ sender, text, ts: new Date().toISOString() });
      }
      return messages;
    "#
}

fn build_zoom_chat_scraper() -> &'static str {
    r#"
      const messages = [];
      let dom = document;
      const iframe = document.querySelector('iframe#webclient');
      if (iframe && iframe.contentDocument) dom = iframe.contentDocument;

      // Zoom Web Client 2025 chat structure:
      // - List items have id starting with "chat-list-item-" or class "new-chat-item__container"
      // - Each item contains author name (in [id^="chat-msg-author"] or .new-chat-item__author)
      // - And message text (in [id^="chat-msg-text"] or .new-chat-message__container__text)
      //
      // Strategy: find each top-level chat item, then extract sender + text
      // from its known sub-structure rather than relying on text scraping.

      // 1. Find all chat list items (top-level only, not nested duplicates)
      const itemSelectors = [
        '[id^="chat-list-item-"]',                    // primary: stable id
        '.new-chat-item__container',                  // primary: known class
        '.new-chat-message__container',               // current web client message node
        '[class*="chat-item-container"]',             // fallback
        '[role="listitem"][class*="chat"]',           // generic fallback
      ];

      let items = [];
      for (const sel of itemSelectors) {
        const found = Array.from(dom.querySelectorAll(sel));
        if (found.length > 0) { items = found; break; }
      }

      // Filter to top-level only: drop items that contain another item
      items = items.filter(el => !items.some(other => other !== el && el.contains(other)));

      for (const item of items) {
        // Skip system messages (UI hints like "Messages addressed to...")
        const itemId = item.id || '';
        if (itemId.includes('system') || itemId.includes('hint')) continue;
        const cls = item.className || '';
        if (typeof cls === 'string' && (cls.includes('system') || cls.includes('hint'))) continue;

        // Extract sender — try several strategies
        let sender = '';
        const authorSelectors = [
          '[id^="chat-msg-author"]',
          '.new-chat-item__author',
          '.chat-item__sender',
          '[class*="chat-message__author"]',
          '[class*="sender-name"]',
        ];
        for (const sel of authorSelectors) {
          const el = item.querySelector(sel);
          if (el && el.textContent?.trim()) { sender = el.textContent.trim(); break; }
        }
        // Fallback: aria-label like "Chat message from John Doe to everyone"
        if (!sender) {
          const aria = item.getAttribute('aria-label') || '';
          const m = aria.match(/from\s+(.+?)(?:\s+to\s+|\s+at\s+|$)/i);
          if (m) sender = m[1].trim();
        }
        if (!sender) sender = 'Participant';

        // Strip role suffix like "John Doe (Host)" → "John Doe"
        sender = sender.replace(/\s*\((Host|Co-host|Me)\)\s*$/i, '').trim();

        // Extract text — try known containers
        let text = '';
        const textSelectors = [
          '[id^="chat-msg-text"]',
          '.new-chat-message__container__text',
          '.chat-rtf-box__display',
          '.new-chat-message__content',
          '.chat-message__text-content',
          '[class*="chat-message__body"]',
          '[class*="chat-msg-text"]',
        ];
        for (const sel of textSelectors) {
          const el = item.querySelector(sel);
          if (el && el.textContent?.trim()) { text = el.textContent.trim(); break; }
        }
        // Fallback: full item text minus the sender prefix
        if (!text && item.textContent) {
          text = item.textContent.trim();
          // Remove leading "SENDER To EVERYONE: " or similar
          if (sender) {
            const prefix = new RegExp("^" + sender.replace(/[.*+?^${}()|[\\]\\\\]/g, "\\\\$&") + "\\s*(?:To\\s+\\S+\\s*)?[:\\s-]+", "i");
            text = text.replace(prefix, '').trim();
          }
        }

        // Skip empty or pure-UI-hint messages
        if (!text) continue;
        if (/^Messages? addressed to/i.test(text)) continue;
        if (/^Direct messages? are private/i.test(text)) continue;

        messages.push({ sender, text, ts: new Date().toISOString() });
      }
      return messages;
    "#
}

// ---------------------------------------------------------------------------
// Provider-specific chat sending (runs inside page.evaluate)
// ---------------------------------------------------------------------------

fn build_google_meet_chat_sender() -> &'static str {
    r#"
      // Open chat panel if not visible
      try {
        const chatBtn = document.querySelector('button[aria-label*="Chat" i], button[aria-label*="chat" i]');
        if (chatBtn) chatBtn.click();
        await new Promise(r => setTimeout(r, 1000));
      } catch {}
      // The actual send uses the Playwright keyboard fallback so Meet receives trusted key events.
      return false;
    "#
}

fn build_teams_chat_sender() -> &'static str {
    r#"
      // Open chat panel if not visible
      try {
        const chatBtn = document.querySelector('button[aria-label*="Chat" i]');
        if (chatBtn) chatBtn.click();
        await new Promise(r => setTimeout(r, 1000));
      } catch {}
      // The actual send uses the Playwright keyboard fallback so Teams receives trusted key events.
      return false;
    "#
}

fn build_zoom_chat_sender() -> &'static str {
    r#"
      let dom = document;
      const iframe = document.querySelector('iframe#webclient');
      if (iframe && iframe.contentDocument) dom = iframe.contentDocument;
      // Open chat panel if needed
      try {
        const chatBtn = Array.from(dom.querySelectorAll('button')).find(b =>
          b.textContent?.toLowerCase().includes('chat') || b.getAttribute('aria-label')?.toLowerCase().includes('chat'));
        if (chatBtn) chatBtn.click();
        await new Promise(r => setTimeout(r, 1000));
      } catch {}
      // The actual send uses the Playwright keyboard fallback so Zoom receives trusted key events.
      return false;
    "#
}

// ---------------------------------------------------------------------------
// STT transcription (reuses Jami pattern)
// ---------------------------------------------------------------------------

pub(crate) fn transcribe_audio_chunk(
    root: &Path,
    audio_path: &Path,
    stt_model: &str,
) -> Result<String> {
    crate::communication::gateway::transcribe_audio_file(root, audio_path, stt_model)
}

// ---------------------------------------------------------------------------
// Meeting invitation detection & scheduling
// ---------------------------------------------------------------------------

/// Known meeting URL patterns.
const MEETING_URL_PATTERNS: &[&str] = &[
    "meet.google.com/",
    "teams.microsoft.com/l/meetup-join/",
    "teams.microsoft.com/meet/",
    "teams.live.com/meet/",
    "zoom.us/j/",
    "zoom.us/my/",
    "zoom.com/j/",
];

/// Extract meeting URLs from a text body (email body, chat message, etc.).
pub(crate) fn extract_meeting_urls(text: &str) -> Vec<String> {
    let mut urls = Vec::new();
    // Simple URL extraction: find https:// followed by known meeting domains
    for word in text.split_whitespace() {
        // Strip common surrounding punctuation/markup
        let candidate = word.trim_matches(|c: char| {
            c == '<'
                || c == '>'
                || c == '"'
                || c == '\''
                || c == '('
                || c == ')'
                || c == '['
                || c == ']'
                || c == ','
                || c == ';'
                || c == '.'
        });
        if !candidate.starts_with("https://") && !candidate.starts_with("http://") {
            continue;
        }
        let candidate = candidate.replace("&amp;", "&");
        let lower = candidate.to_lowercase();
        if MEETING_URL_PATTERNS.iter().any(|pat| lower.contains(pat)) {
            // Normalize: strip trailing fragments and tracking params
            let clean = candidate
                .split('#')
                .next()
                .unwrap_or(candidate.as_str())
                .to_string();
            if !urls.contains(&clean) {
                urls.push(clean);
            }
        }
    }
    urls
}

/// Parse a meeting time from ICS-style DTSTART or common date patterns.
/// Returns ISO 8601 timestamp if found.
pub(crate) fn extract_meeting_time_from_text(text: &str) -> Option<String> {
    extract_ics_value(text, "DTSTART").and_then(|value| parse_ics_datetime(&value))
}

fn extract_meeting_end_time_from_text(text: &str) -> Option<String> {
    extract_ics_value(text, "DTEND").and_then(|value| parse_ics_datetime(&value))
}

fn extract_meeting_uid_from_text(text: &str) -> Option<String> {
    extract_ics_value(text, "UID")
}

fn extract_meeting_sequence_from_text(text: &str) -> Option<String> {
    extract_ics_value(text, "SEQUENCE")
}

fn extract_meeting_summary_from_text(text: &str) -> Option<String> {
    extract_ics_value(text, "SUMMARY")
}

fn extract_meeting_method_from_text(text: &str) -> Option<String> {
    extract_ics_value(text, "METHOD").map(|value| value.to_ascii_uppercase())
}

fn extract_ics_value(text: &str, field: &str) -> Option<String> {
    let needle = field.to_ascii_uppercase();
    for line in unfold_ics_lines(text) {
        let trimmed = line.trim();
        let upper = trimmed.to_ascii_uppercase();
        if upper == needle
            || upper.starts_with(&(needle.clone() + ":"))
            || upper.starts_with(&(needle.clone() + ";"))
        {
            let value = trimmed.rsplit_once(':')?.1.trim();
            if !value.is_empty() {
                return Some(unescape_ics_text(value));
            }
        }
    }
    None
}

fn unfold_ics_lines(text: &str) -> Vec<String> {
    let mut lines: Vec<String> = Vec::new();
    for raw in text.lines() {
        if raw.starts_with(' ') || raw.starts_with('\t') {
            if let Some(last) = lines.last_mut() {
                last.push_str(raw.trim_start());
            }
        } else {
            lines.push(raw.trim_end_matches('\r').to_string());
        }
    }
    lines
}

fn unescape_ics_text(value: &str) -> String {
    value
        .replace("\\n", "\n")
        .replace("\\N", "\n")
        .replace("\\,", ",")
        .replace("\\;", ";")
        .replace("\\\\", "\\")
}

fn parse_ics_datetime(value: &str) -> Option<String> {
    let value = value.trim();
    if value.len() < 15 {
        return None;
    }
    let year = value.get(0..4)?;
    let month = value.get(4..6)?;
    let day = value.get(6..8)?;
    let hour = value.get(9..11)?;
    let min = value.get(11..13)?;
    let sec = value.get(13..15)?;
    let tz = if value.ends_with('Z') { "Z" } else { "" };
    Some(format!("{year}-{month}-{day}T{hour}:{min}:{sec}{tz}"))
}

/// Detect whether an email body indicates a meeting cancellation.
pub(crate) fn is_meeting_cancellation(subject: &str, body: &str) -> bool {
    let lower_subject = subject.to_lowercase();
    let lower_body = body.to_lowercase();
    // Common cancellation indicators
    lower_subject.contains("canceled")
        || lower_subject.contains("cancelled")
        || lower_subject.contains("abgesagt")
        || lower_body.contains("has been canceled")
        || lower_body.contains("has been cancelled")
        || lower_body.contains("meeting wurde abgesagt")
        || lower_body.contains("method:cancel")
        || extract_meeting_method_from_text(body).as_deref() == Some("CANCEL")
}

/// Detect whether an email body indicates a meeting time change.
pub(crate) fn is_meeting_update(subject: &str, body: &str) -> bool {
    let lower_subject = subject.to_lowercase();
    let lower_body = body.to_lowercase();
    lower_subject.contains("updated")
        || lower_subject.contains("rescheduled")
        || lower_subject.contains("aktualisiert")
        || lower_subject.contains("verschoben")
        || lower_body.contains("has been updated")
        || lower_body.contains("has been rescheduled")
        || lower_body.contains("new time:")
        || lower_body.contains("neue zeit:")
}

/// Build a cron expression for a one-shot meeting at a specific ISO timestamp.
/// Cron format: minute hour day month *
/// The schedule module fires when `next_run_at <= now`, so we set it directly.
pub(crate) fn cron_for_meeting_time(iso_time: &str) -> Option<String> {
    // Parse "2026-04-15T14:00:00Z" → minute=0 hour=14 day=15 month=4
    if iso_time.len() < 16 {
        return None;
    }
    let month: u32 = iso_time[5..7].parse().ok()?;
    let day: u32 = iso_time[8..10].parse().ok()?;
    let hour: u32 = iso_time[11..13].parse().ok()?;
    let min: u32 = iso_time[14..16].parse().ok()?;
    Some(format!("{min} {hour} {day} {month} *"))
}

/// Unique schedule name for a meeting URL (stable across updates).
pub(crate) fn meeting_schedule_name(meeting_url: &str) -> String {
    format!("meeting-join:{}", stable_digest(meeting_url))
}

fn meeting_schedule_name_for_invitation(meeting_url: &str, uid: Option<&str>) -> String {
    if let Some(stable_key) = uid.map(str::trim).filter(|value| !value.is_empty()) {
        return format!("meeting-join:{}", stable_digest(stable_key));
    }
    meeting_schedule_name(meeting_url)
}

fn meeting_join_time(meeting_time_iso: &str) -> String {
    DateTime::parse_from_rfc3339(meeting_time_iso)
        .map(|dt| (dt.with_timezone(&Utc) - Duration::minutes(1)).to_rfc3339())
        .unwrap_or_else(|_| meeting_time_iso.to_string())
}

/// Schedule a meeting join via the CTOX schedule system.
/// Creates or updates a scheduled task that will fire at the meeting start time.
pub(crate) fn schedule_meeting_join(
    root: &Path,
    meeting_url: &str,
    meeting_time_iso: &str,
    bot_name: &str,
) -> Result<Value> {
    schedule_meeting_join_with_metadata(
        root,
        meeting_url,
        meeting_time_iso,
        bot_name,
        None,
        None,
        None,
    )
}

fn schedule_meeting_join_with_metadata(
    root: &Path,
    meeting_url: &str,
    meeting_time_iso: &str,
    bot_name: &str,
    uid: Option<&str>,
    sequence: Option<&str>,
    summary: Option<&str>,
) -> Result<Value> {
    let provider =
        MeetingProvider::detect(meeting_url).context("cannot detect meeting provider from URL")?;
    let join_time_iso = meeting_join_time(meeting_time_iso);
    let cron_expr = cron_for_meeting_time(&join_time_iso)
        .context("cannot parse meeting time into cron expression")?;
    let schedule_name = meeting_schedule_name_for_invitation(meeting_url, uid);
    let thread_key = format!("meeting:{}", provider.as_str());

    let payload = json!({
        "url": meeting_url,
        "bot_name": bot_name,
        "provider": provider.as_str(),
        "meeting_time": meeting_time_iso,
        "join_time": join_time_iso,
        "uid": uid,
        "sequence": sequence,
        "summary": summary,
    });
    let prompt = format!(
        "CTOX_MEETING_JOIN: {payload}\n\
         Join the {provider} meeting at {url} as \"{bot_name}\". \
         Capture audio transcript and monitor chat. \
         If no other participants join within 15 minutes, leave the meeting. \
         After the meeting ends, summarize the transcript and create tickets. \
         Create durable knowledge only when the meeting produced reusable operational procedure; \
         durable knowledge must be a Skillbook/Runbook/Runbook-Item, not a ticket_knowledge_entries note.",
        provider = provider.as_str(),
        url = meeting_url,
        bot_name = bot_name,
    );

    let request = crate::mission::schedule::ScheduleEnsureRequest {
        name: schedule_name.clone(),
        cron_expr,
        prompt,
        thread_key,
        skill: Some("system-onboarding".to_string()),
    };
    let task = crate::mission::schedule::ensure_task(root, request)?;

    // Also persist the meeting details for the join logic
    let sessions_dir = meeting_sessions_dir(root);
    fs::create_dir_all(&sessions_dir)?;
    let session_file = sessions_dir.join(format!("{}.json", schedule_name));
    let session_meta = json!({
        "schedule_name": schedule_name,
        "meeting_url": meeting_url,
        "meeting_time": meeting_time_iso,
        "join_time": join_time_iso,
        "provider": provider.as_str(),
        "bot_name": bot_name,
        "uid": uid,
        "sequence": sequence,
        "summary": summary,
        "status": "scheduled",
        "created_at": now_iso_string(),
    });
    fs::write(&session_file, serde_json::to_string_pretty(&session_meta)?)?;

    Ok(json!({
        "ok": true,
        "action": "scheduled",
        "schedule_name": schedule_name,
        "task_id": task.task_id,
        "meeting_url": meeting_url,
        "meeting_time": meeting_time_iso,
        "join_time": join_time_iso,
        "provider": provider.as_str(),
        "cron_expr": task.cron_expr,
        "next_run_at": task.next_run_at,
    }))
}

/// Cancel a scheduled meeting join.
pub(crate) fn cancel_meeting_join(root: &Path, meeting_url: &str) -> Result<Value> {
    cancel_meeting_join_with_uid(root, meeting_url, None)
}

fn cancel_meeting_join_with_uid(
    root: &Path,
    meeting_url: &str,
    uid: Option<&str>,
) -> Result<Value> {
    let schedule_name = meeting_schedule_name_for_invitation(meeting_url, uid);
    let session_file = meeting_sessions_dir(root).join(format!("{schedule_name}.json"));
    let provider_thread_key = MeetingProvider::detect(meeting_url)
        .map(|provider| format!("meeting:{}", provider.as_str()));

    // Remove matching scheduled tasks by persisted metadata instead of relying
    // on reconstructing the schedule module's task-id derivation.
    if let Ok(tasks) = crate::mission::schedule::list_tasks(root) {
        for task in tasks {
            let provider_matches = provider_thread_key
                .as_deref()
                .map(|thread_key| task.thread_key == thread_key)
                .unwrap_or(true);
            if task.name == schedule_name && provider_matches {
                if let Err(err) = crate::mission::schedule::remove_task(root, &task.task_id) {
                    eprintln!(
                        "note: could not remove scheduled task {}: {err}",
                        task.task_id
                    );
                }
            }
        }
    }

    // Update session file
    if session_file.exists() {
        let _ = fs::write(
            &session_file,
            serde_json::to_string_pretty(&json!({
                "schedule_name": schedule_name,
                "meeting_url": meeting_url,
                "status": "cancelled",
                "cancelled_at": now_iso_string(),
            }))?,
        );
    }

    Ok(json!({
        "ok": true,
        "action": "cancelled",
        "schedule_name": schedule_name,
        "meeting_url": meeting_url,
    }))
}

/// Process an inbound email to detect meeting invitations, updates, or cancellations.
/// Returns a summary of actions taken.
pub(crate) fn process_email_for_meetings(
    root: &Path,
    subject: &str,
    body: &str,
    bot_name: &str,
) -> Result<Value> {
    let urls = extract_meeting_urls(body);
    if urls.is_empty() {
        return Ok(json!({"ok": true, "action": "none", "reason": "no meeting URLs found"}));
    }

    let mut results = Vec::new();
    let uid = extract_meeting_uid_from_text(body);
    let sequence = extract_meeting_sequence_from_text(body);
    let summary = extract_meeting_summary_from_text(body);
    let meeting_time = extract_meeting_time_from_text(body);
    let meeting_end_time = extract_meeting_end_time_from_text(body);

    for url in &urls {
        if is_meeting_cancellation(subject, body) {
            let result = cancel_meeting_join_with_uid(root, url, uid.as_deref())?;
            results.push(result);
            continue;
        }

        if let Some(ref time) = meeting_time {
            if is_meeting_update(subject, body) {
                // Update = cancel old + schedule new
                let _ = cancel_meeting_join_with_uid(root, url, uid.as_deref());
            }
            let mut result = schedule_meeting_join_with_metadata(
                root,
                url,
                time,
                bot_name,
                uid.as_deref(),
                sequence.as_deref(),
                summary.as_deref(),
            )?;
            if let Some(object) = result.as_object_mut() {
                object.insert("uid".to_string(), json!(uid));
                object.insert("sequence".to_string(), json!(sequence));
                object.insert("summary".to_string(), json!(summary));
                object.insert("meeting_end_time".to_string(), json!(meeting_end_time));
            }
            results.push(result);
        } else {
            results.push(json!({
                "ok": false,
                "meeting_url": url,
                "uid": uid,
                "reason": "meeting URL found but no start time detected",
            }));
        }
    }

    let successful_results = results
        .iter()
        .filter(|result| {
            result
                .get("ok")
                .and_then(serde_json::Value::as_bool)
                .unwrap_or(false)
        })
        .count();
    let action = if results.is_empty() {
        "none"
    } else if successful_results > 0 {
        "processed"
    } else {
        "needs_review"
    };

    Ok(json!({
        "ok": true,
        "action": action,
        "results": results,
    }))
}

// ---------------------------------------------------------------------------
// Meeting join timeout & lifecycle
// ---------------------------------------------------------------------------

// Meeting join behavior constants are now embedded directly in the
// Playwright runner templates (participant detection + silence detection).

/// Update the Playwright runner template's participant monitoring to include
/// the empty-meeting timeout. This is already embedded in the runner script
/// via the participant count monitoring interval — when count <= 1 for 60s
/// the meeting ends. The 15-minute initial empty timeout is handled by
/// injecting it into the runner script.
///
/// Build the runner script with integrated timeout/inactivity detection.
/// Since the transplanted recording template already includes participant-detection,
/// silence-detection, and max-duration timeouts from the reference implementation,
/// this is now a thin wrapper around `build_meeting_runner_script`.
pub(crate) fn build_meeting_runner_script_with_timeout(
    config: &MeetingSessionConfig,
) -> Result<String> {
    // The recording template now natively handles:
    //  - Google/Zoom: participant count detection (6 methods), AudioContext silence detection
    //  - Teams: `parec` audio silence detection, participant count via aria-label
    //  - All providers: max duration timeout
    build_meeting_runner_script(config)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn meeting_sessions_dir(root: &Path) -> PathBuf {
    communication_runtime::channel_dir(root, "meeting").join("sessions")
}

fn legacy_meeting_sessions_dir(root: &Path) -> PathBuf {
    root.join("runtime").join("meeting_sessions")
}

fn existing_meeting_session_dirs(root: &Path) -> Vec<PathBuf> {
    let canonical = meeting_sessions_dir(root);
    let legacy = legacy_meeting_sessions_dir(root);
    let mut dirs = Vec::new();
    if canonical.exists() {
        dirs.push(canonical.clone());
    }
    if legacy.exists() && legacy != canonical {
        dirs.push(legacy);
    }
    dirs
}

fn meeting_session_file(root: &Path, session_id: &str) -> PathBuf {
    let canonical = meeting_sessions_dir(root).join(format!("{session_id}.json"));
    if canonical.exists() {
        return canonical;
    }
    let legacy = legacy_meeting_sessions_dir(root).join(format!("{session_id}.json"));
    if legacy.exists() {
        return legacy;
    }
    canonical
}

fn meeting_session_artifact_file(root: &Path, session_id: &str, suffix: &str) -> PathBuf {
    let canonical = meeting_sessions_dir(root).join(format!("{session_id}{suffix}"));
    if canonical.exists() {
        return canonical;
    }
    let legacy = legacy_meeting_sessions_dir(root).join(format!("{session_id}{suffix}"));
    if legacy.exists() {
        return legacy;
    }
    canonical
}

/// Load the final artifacts of a meeting session: the session metadata
/// JSON (speakers, duration, provider, etc.), the full STT transcript,
/// and the captured chat log. Returns a structured JSON suitable for
/// direct emission by the `ctox meeting transcript` CLI and consumption
/// by the agent-runtime `meeting_get_transcript` tool.
///
/// Missing transcript/chatlog files are returned as empty strings rather
/// than errors — an active session may have metadata persisted before
/// finalize_meeting_session has written the text files.
pub(crate) fn load_meeting_transcript(root: &Path, session_id: &str) -> Result<Value> {
    let session_path = meeting_session_file(root, session_id);
    let session: Value = if session_path.exists() {
        let contents = fs::read_to_string(&session_path)
            .with_context(|| format!("read meeting session {}", session_path.display()))?;
        let session: Value = serde_json::from_str(&contents)
            .with_context(|| format!("parse meeting session JSON at {}", session_path.display()))?;
        MeetingSessionStatus::from_session_value(&session).with_context(|| {
            format!(
                "invalid meeting session status at {}",
                session_path.display()
            )
        })?;
        session
    } else {
        anyhow::bail!("no meeting session found with id {session_id}");
    };

    let transcript_path = meeting_session_artifact_file(root, session_id, "-transcript.txt");
    let chatlog_path = meeting_session_artifact_file(root, session_id, "-chatlog.txt");

    let transcript = fs::read_to_string(&transcript_path).unwrap_or_default();
    let chatlog = fs::read_to_string(&chatlog_path).unwrap_or_default();

    Ok(json!({
        "ok": true,
        "session_id": session_id,
        "session": session,
        "transcript": transcript,
        "chatlog": chatlog,
        "transcript_path": transcript_path.display().to_string(),
        "chatlog_path": chatlog_path.display().to_string(),
    }))
}

fn now_iso_string() -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = now.as_secs();
    let nanos = now.subsec_nanos();
    // Simple ISO-8601 without external deps
    let (year, month, day, hour, min, sec) = epoch_to_datetime(secs);
    format!(
        "{year:04}-{month:02}-{day:02}T{hour:02}:{min:02}:{sec:02}.{millis:03}Z",
        millis = nanos / 1_000_000
    )
}

fn now_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

fn epoch_to_datetime(epoch_secs: u64) -> (u64, u64, u64, u64, u64, u64) {
    let days = epoch_secs / 86400;
    let time = epoch_secs % 86400;
    let hour = time / 3600;
    let min = (time % 3600) / 60;
    let sec = time % 60;

    // Simplified date calculation (accurate for 1970-2099)
    let mut y = 1970u64;
    let mut remaining_days = days;
    loop {
        let days_in_year =
            if y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400)) {
                366
            } else {
                365
            };
        if remaining_days < days_in_year {
            break;
        }
        remaining_days -= days_in_year;
        y += 1;
    }
    let leap = y.is_multiple_of(4) && (!y.is_multiple_of(100) || y.is_multiple_of(400));
    let month_days: [u64; 12] = [
        31,
        if leap { 29 } else { 28 },
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ];
    let mut m = 0usize;
    while m < 12 && remaining_days >= month_days[m] {
        remaining_days -= month_days[m];
        m += 1;
    }
    (y, (m + 1) as u64, remaining_days + 1, hour, min, sec)
}

fn stable_digest(input: &str) -> String {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(prefix: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!(
            "ctox-meeting-{prefix}-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).expect("temp root");
        root
    }

    #[test]
    fn detect_meeting_provider_from_url() {
        assert_eq!(
            MeetingProvider::detect("https://meet.google.com/abc-defg-hij"),
            Some(MeetingProvider::GoogleMeet)
        );
        assert_eq!(
            MeetingProvider::detect("https://teams.microsoft.com/l/meetup-join/abc"),
            Some(MeetingProvider::MicrosoftTeams)
        );
        assert_eq!(
            MeetingProvider::detect("https://us04web.zoom.us/j/123456"),
            Some(MeetingProvider::Zoom)
        );
        assert_eq!(MeetingProvider::detect("https://example.com"), None);
    }

    #[test]
    fn mention_detection_is_case_insensitive() {
        assert!(MeetingSession::is_mention("Hey @CTOX what do you think?"));
        assert!(MeetingSession::is_mention("@ctox please summarize"));
        assert!(MeetingSession::is_mention("Hello @Ctox!"));
        assert!(!MeetingSession::is_mention("This is a normal message"));
    }

    #[test]
    fn mention_detection_respects_word_boundary() {
        // Bot's full display name in chat headers should still match
        assert!(MeetingSession::is_mention(
            "@INF Yoda Notetaker hat geantwortet"
        ));
        assert!(MeetingSession::is_mention("@ctox notetaker"));
        // But unrelated tokens that contain "ctox" as a substring should not
        assert!(!MeetingSession::is_mention("@ctoxbar"));
        assert!(!MeetingSession::is_mention("@ctoxology is a fake word"));
        // Embedded inside a longer URL or token also shouldn't match
        assert!(!MeetingSession::is_mention(
            "https://example.com/@ctoxapi/foo"
        ));
    }

    #[test]
    fn engine_reachable_check_returns_false_for_closed_port() {
        assert!(!check_engine_reachable(Path::new(
            "/definitely/not/a/ctox/root"
        )));
    }

    #[test]
    fn linux_meeting_runner_uses_xvfb_when_display_is_missing() {
        assert_eq!(
            should_wrap_browser_runner_with_xvfb(None),
            cfg!(target_os = "linux")
        );
        assert!(!should_wrap_browser_runner_with_xvfb(Some(OsStr::new(
            ":99"
        ))));
        assert_eq!(
            should_wrap_browser_runner_with_xvfb(Some(OsStr::new(""))),
            cfg!(target_os = "linux")
        );
        assert!(MEETING_XVFB_SERVER_ARGS.contains("1920x1080x24"));
    }

    #[test]
    fn runner_exit_persists_ended_status_and_reason() {
        let root = temp_root("runner-exit");
        let config = MeetingSessionConfig {
            root: root.clone(),
            meeting_url: "https://zoom.us/j/123".to_string(),
            provider: MeetingProvider::Zoom,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 60,
            audio_chunk_seconds: 30,
            stt_model: String::new(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let mut session = MeetingSession::new(&config);
        session.status = MeetingSessionStatus::Running;
        record_runner_exit(
            &root,
            &mut session,
            "runner_exit_status:signal: 9".to_string(),
        )
        .expect("persist runner exit");

        let persisted: Value = serde_json::from_str(
            &fs::read_to_string(meeting_session_file(&root, &session.session_id)).unwrap(),
        )
        .unwrap();
        assert_eq!(persisted["status"], "ended");
        assert_eq!(persisted["end_reason"], "runner_exit_status:signal: 9");
        assert!(persisted["ended_at"].as_str().is_some());
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn meeting_session_status_rejects_unknown_value() {
        let err = MeetingSessionStatus::parse("teleported").expect_err("unknown status");
        assert_eq!(
            err.to_string(),
            "unknown meeting session status `teleported`"
        );
    }

    #[test]
    fn self_loop_protection_filters_bot_messages() {
        let config = MeetingSessionConfig {
            root: PathBuf::from("/tmp"),
            meeting_url: "https://zoom.us/j/123".to_string(),
            provider: MeetingProvider::Zoom,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 60,
            audio_chunk_seconds: 30,
            stt_model: String::new(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let mut session = MeetingSession::new(&config);

        // Sender contains bot name → own message
        assert!(session.is_own_message("INF Yoda Notetaker", "hello"));
        assert!(session.is_own_message("INF Yoda Notetaker (Host)", "hello"));
        assert!(session.is_own_message("ctox notetaker", "hello"));

        // Real participants are not filtered
        assert!(!session.is_own_message("Michael Welsch", "@CTOX hello"));
        assert!(!session.is_own_message("Participant", "regular message"));

        // Addressing the bot by name is an inbound mention, not self-loop.
        assert!(!session.is_own_message("Participant", "INF Yoda Notetaker: I heard you"));
        assert!(MeetingSession::is_mention(
            "INF Yoda Notetaker: I heard you"
        ));
        assert!(MeetingSession::is_mention(
            "INF\u{00a0}Yoda\u{00a0}Notetaker: bitte pruefen"
        ));
        session
            .outbound_chat_texts
            .push("CTOX Test: Chat-Bridge aktiv.".to_string());
        assert!(session.is_own_message("You", "You20:35CNCTOX Test: Chat-Bridge aktiv."));
        assert!(session.is_own_message("Participant", "You20:35CNCTOX Test: Chat-Bridge aktiv."));

        // Text mentions bot name but doesn't start with it (real user message)
        assert!(!session.is_own_message("Michael", "Hey @INF Yoda Notetaker what do you think?"));
    }

    #[test]
    fn session_roundtrip_json() {
        let config = MeetingSessionConfig {
            root: PathBuf::from("/tmp/test"),
            meeting_url: "https://meet.google.com/abc".to_string(),
            provider: MeetingProvider::GoogleMeet,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 180,
            audio_chunk_seconds: 30,
            stt_model: String::new(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let session = MeetingSession::new(&config);
        let json = session.to_json();
        assert_eq!(json["provider"], "google");
        assert_eq!(json["status"], "joining");
        assert_eq!(json["bot_name"], "INF Yoda Notetaker");
        assert_eq!(json["transcript_segment_count"], 0);
        assert_eq!(json["speaker_signal_count"], 0);
    }

    #[test]
    fn meeting_runtime_defaults_to_voxtral_4b_stt() {
        let config = MeetingSessionConfig::from_runtime(
            Path::new("/tmp"),
            "https://meet.google.com/abc-defg-hij",
            &BTreeMap::new(),
        )
        .expect("meeting config");
        assert_eq!(config.stt_model, DEFAULT_MEETING_STT_MODEL);
    }

    #[test]
    fn meeting_runtime_replaces_legacy_stt_models_with_voxtral_4b() {
        let mut runtime = BTreeMap::new();
        runtime.insert("CTOX_STT_MODEL".to_string(), "legacy-stt-model".to_string());
        let config = MeetingSessionConfig::from_runtime(
            Path::new("/tmp"),
            "https://zoom.us/j/123456789",
            &runtime,
        )
        .expect("meeting config");
        assert_eq!(config.stt_model, DEFAULT_MEETING_STT_MODEL);
    }

    #[test]
    fn transcript_segments_render_speaker_source_and_confidence() {
        let config = MeetingSessionConfig {
            root: PathBuf::from("/tmp/test"),
            meeting_url: "https://meet.google.com/abc".to_string(),
            provider: MeetingProvider::GoogleMeet,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 180,
            audio_chunk_seconds: 30,
            stt_model: DEFAULT_MEETING_STT_MODEL.to_string(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let mut session = MeetingSession::new(&config);
        session.push_platform_transcript(TranscriptSegment {
            timestamp: "2026-04-28T12:00:00Z".to_string(),
            speaker_display: "Alice".to_string(),
            speaker_id: Some("alice-platform-id".to_string()),
            source: "platform_caption".to_string(),
            confidence: 0.9,
            text: "The rollout is blocked by permissions.".to_string(),
        });

        let transcript = session.full_transcript();
        assert!(transcript.contains("Alice: The rollout is blocked"));
        assert!(transcript.contains("source=platform_caption"));
        assert!(transcript.contains("confidence=0.90"));

        let snapshot = session_transcript_snapshot(&session.to_json(), 12);
        assert!(snapshot.contains("Alice: The rollout is blocked"));
    }

    #[test]
    fn stt_segments_use_active_speaker_when_available() {
        let config = MeetingSessionConfig {
            root: PathBuf::from("/tmp/test"),
            meeting_url: "https://zoom.us/j/123456".to_string(),
            provider: MeetingProvider::Zoom,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 180,
            audio_chunk_seconds: 30,
            stt_model: DEFAULT_MEETING_STT_MODEL.to_string(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let mut session = MeetingSession::new(&config);
        let signal = SpeakerSignal {
            timestamp: "2026-04-28T12:00:00Z".to_string(),
            speaker_display: "Bob".to_string(),
            speaker_id: None,
            source: "platform_active_speaker".to_string(),
            confidence: 0.6,
        };
        session.push_stt_transcript(
            "I can take the deployment ticket.".to_string(),
            Some(&signal),
        );
        assert_eq!(
            session.transcript_segments[0].source,
            "stt_with_active_speaker"
        );
        assert_eq!(session.transcript_segments[0].speaker_display, "Bob");
        assert!(session.full_transcript().contains("Bob: I can take"));
    }

    #[test]
    fn full_transcript_keeps_stt_when_platform_captions_exist() {
        let config = MeetingSessionConfig {
            root: PathBuf::from("/tmp/test"),
            meeting_url: "https://teams.microsoft.com/meet/demo".to_string(),
            provider: MeetingProvider::MicrosoftTeams,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 180,
            audio_chunk_seconds: 30,
            stt_model: DEFAULT_MEETING_STT_MODEL.to_string(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let mut session = MeetingSession::new(&config);
        session.push_platform_transcript(TranscriptSegment {
            timestamp: "2026-04-28T12:00:00Z".to_string(),
            speaker_display: "Teams".to_string(),
            speaker_id: None,
            source: "platform_caption".to_string(),
            confidence: 0.8,
            text: "Screen shared.".to_string(),
        });
        session.push_stt_transcript(
            "A participant described the Salesforce assignment workflow.".to_string(),
            None,
        );

        let transcript = session.full_transcript();
        assert!(transcript.contains("Teams: Screen shared."));
        assert!(transcript.contains("Salesforce assignment workflow"));
        assert!(transcript.contains("source=stt"));
    }

    #[test]
    fn recording_fallback_is_needed_only_without_usable_stt() {
        let config = MeetingSessionConfig {
            root: PathBuf::from("/tmp/test"),
            meeting_url: "https://teams.microsoft.com/meet/demo".to_string(),
            provider: MeetingProvider::MicrosoftTeams,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 180,
            audio_chunk_seconds: 30,
            stt_model: DEFAULT_MEETING_STT_MODEL.to_string(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let mut session = MeetingSession::new(&config);
        session.push_platform_transcript(TranscriptSegment {
            timestamp: "2026-04-28T12:00:00Z".to_string(),
            speaker_display: "Teams".to_string(),
            speaker_id: None,
            source: "platform_caption".to_string(),
            confidence: 0.8,
            text: "You are screen sharing.".to_string(),
        });
        assert!(meeting_transcript_needs_recording_fallback(&session));

        session.push_stt_transcript("A participant gave a real update.".to_string(), None);
        assert!(!meeting_transcript_needs_recording_fallback(&session));
    }

    #[test]
    fn full_recording_candidates_ignore_audio_chunks_and_prefer_largest() {
        let root = temp_root("recording-candidates");
        let session_id = "meeting-microsoft-recording-test";
        let sessions_dir = meeting_sessions_dir(&root);
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
        std::fs::write(
            sessions_dir.join(format!("{session_id}-manual-recording.mp4")),
            vec![0; 16],
        )
        .expect("manual recording");
        std::fs::write(
            sessions_dir.join(format!("{session_id}-recording.mp4")),
            vec![0; 32],
        )
        .expect("recording");
        let audio_dir = sessions_dir.join(format!("{session_id}-audio"));
        std::fs::create_dir_all(&audio_dir).expect("audio dir");
        std::fs::write(audio_dir.join("chunk-001.webm"), vec![0; 64]).expect("chunk");

        let candidates = full_meeting_recording_candidates(&root, session_id);
        assert_eq!(candidates.len(), 2);
        assert!(candidates[0]
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap()
            .ends_with("-recording.mp4"));
        assert!(candidates.iter().all(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .map(|name| name.contains("recording"))
                .unwrap_or(false)
        }));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn build_runner_script_compiles_for_all_providers() {
        for provider in [
            MeetingProvider::GoogleMeet,
            MeetingProvider::MicrosoftTeams,
            MeetingProvider::Zoom,
        ] {
            let config = MeetingSessionConfig {
                root: PathBuf::from("/tmp"),
                meeting_url: "https://example.com".to_string(),
                provider,
                bot_name: "Test Bot".to_string(),
                max_duration_minutes: 60,
                audio_chunk_seconds: 30,
                stt_model: String::new(),
                realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
                mistral_api_key: None,
            };
            let script = build_meeting_runner_script(&config).unwrap();
            assert!(script.contains("chromium"));
            assert!(script.contains("emit"));
        }
    }

    #[test]
    fn extract_meeting_urls_from_email_body() {
        let body =
            "Hi team,\n\nJoin the meeting here: https://meet.google.com/abc-defg-hij\n\nThanks";
        let urls = extract_meeting_urls(body);
        assert_eq!(urls, vec!["https://meet.google.com/abc-defg-hij"]);

        let body2 = "Teams meeting: https://teams.microsoft.com/l/meetup-join/abc123 and also https://zoom.us/j/999";
        let urls2 = extract_meeting_urls(body2);
        assert_eq!(urls2.len(), 2);
        assert!(urls2[0].contains("teams.microsoft.com"));
        assert!(urls2[1].contains("zoom.us"));

        let body3 = "No meeting links here, just regular text.";
        assert!(extract_meeting_urls(body3).is_empty());
    }

    #[test]
    fn extract_meeting_time_from_ics() {
        let ics =
            "BEGIN:VCALENDAR\nDTSTART:20260415T140000Z\nDTEND:20260415T150000Z\nEND:VCALENDAR";
        assert_eq!(
            extract_meeting_time_from_text(ics),
            Some("2026-04-15T14:00:00Z".to_string())
        );

        let no_time = "Just a regular email body without ICS data.";
        assert_eq!(extract_meeting_time_from_text(no_time), None);
    }

    #[test]
    fn extracts_uid_sequence_summary_and_folded_ics_values() {
        let ics = "BEGIN:VCALENDAR\nUID:meeting-123@example.com\nSEQUENCE:4\nSUMMARY:Weekly\\, Platform Review\nDTSTART;TZID=Europe/Berlin:20260415T140000\nEND:VCALENDAR\n";
        assert_eq!(
            extract_meeting_uid_from_text(ics).as_deref(),
            Some("meeting-123@example.com")
        );
        assert_eq!(
            extract_meeting_sequence_from_text(ics).as_deref(),
            Some("4")
        );
        assert_eq!(
            extract_meeting_summary_from_text(ics).as_deref(),
            Some("Weekly, Platform Review")
        );
        assert_eq!(
            extract_meeting_time_from_text(ics).as_deref(),
            Some("2026-04-15T14:00:00")
        );
    }

    #[test]
    fn meeting_schedule_name_prefers_calendar_uid() {
        let url_a = "https://meet.google.com/aaa-bbbb-ccc";
        let url_b = "https://meet.google.com/xxx-yyyy-zzz";
        assert_eq!(
            meeting_schedule_name_for_invitation(url_a, Some("uid-1")),
            meeting_schedule_name_for_invitation(url_b, Some("uid-1"))
        );
        assert_ne!(
            meeting_schedule_name_for_invitation(url_a, None),
            meeting_schedule_name_for_invitation(url_b, None)
        );
    }

    #[test]
    fn process_email_with_link_but_no_time_needs_review() {
        let root = temp_root("no-time");
        let _ = std::fs::remove_dir_all(&root);
        let result = process_email_for_meetings(
            &root,
            "Meeting invitation",
            "Join here: https://meet.google.com/abc-defg-hij",
            "INF Yoda Notetaker",
        )
        .expect("meeting parse result");
        assert_eq!(result["action"], "needs_review");
        assert_eq!(result["results"][0]["ok"], false);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn process_email_request_schedules_join_task_with_marker_and_uid() {
        let root = temp_root("schedule-request");
        let body = "BEGIN:VCALENDAR\nMETHOD:REQUEST\nUID:uid-request-1@example.com\nSEQUENCE:2\nSUMMARY:Platform Review\nDTSTART:20260428T130000Z\nDTEND:20260428T133000Z\nDESCRIPTION:Join https://meet.google.com/abc-defg-hij\nEND:VCALENDAR";
        let result = process_email_for_meetings(
            &root,
            "Invitation: Platform Review",
            body,
            "INF Yoda Notetaker",
        )
        .expect("schedule result");
        assert_eq!(result["action"], "processed");
        assert_eq!(result["results"][0]["action"], "scheduled");
        assert_eq!(result["results"][0]["uid"], "uid-request-1@example.com");

        let tasks = crate::mission::schedule::list_tasks(&root).expect("scheduled tasks");
        assert_eq!(tasks.len(), 1);
        assert!(tasks[0].prompt.starts_with("CTOX_MEETING_JOIN:"));
        assert!(tasks[0]
            .prompt
            .contains("https://meet.google.com/abc-defg-hij"));
        assert_eq!(tasks[0].cron_expr, "59 12 28 4 *");
        assert_eq!(
            result["results"][0]["join_time"],
            "2026-04-28T12:59:00+00:00"
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn process_email_cancel_removes_uid_based_schedule() {
        let root = temp_root("schedule-cancel");
        let request_body = "BEGIN:VCALENDAR\nMETHOD:REQUEST\nUID:uid-cancel-1@example.com\nDTSTART:20260428T130000Z\nDESCRIPTION:Join https://zoom.us/j/123456789\nEND:VCALENDAR";
        process_email_for_meetings(
            &root,
            "Invitation: Standup",
            request_body,
            "INF Yoda Notetaker",
        )
        .expect("schedule result");
        assert_eq!(
            crate::mission::schedule::list_tasks(&root)
                .expect("scheduled tasks")
                .len(),
            1
        );

        let cancel_body = "BEGIN:VCALENDAR\nMETHOD:CANCEL\nUID:uid-cancel-1@example.com\nDESCRIPTION:Join https://zoom.us/j/123456789\nEND:VCALENDAR";
        let cancel = process_email_for_meetings(
            &root,
            "Meeting cancelled: Standup",
            cancel_body,
            "INF Yoda Notetaker",
        )
        .expect("cancel result");
        assert_eq!(cancel["action"], "processed");
        assert_eq!(cancel["results"][0]["action"], "cancelled");
        assert!(crate::mission::schedule::list_tasks(&root)
            .expect("scheduled tasks")
            .is_empty());
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sync_sends_first_mention_ack_once_and_marks_priority() {
        let root = temp_root("mention-ack");
        let sessions_dir = meeting_sessions_dir(&root);
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
        let db_path = root.join("runtime/ctox.sqlite3");
        let stdin_path = sessions_dir.join("session-1.stdin");
        std::fs::write(&stdin_path, "").expect("stdin file");
        let session_path = sessions_dir.join("session-1.json");
        std::fs::write(
            &session_path,
            serde_json::to_string_pretty(&json!({
                "session_id": "session-1",
                "provider": "google",
                "status": "active",
                "stdin_pipe": stdin_path.display().to_string(),
                "transcript_chunks": [
                    "Alice said the rollout is blocked by permissions.",
                    "Bob offered to prepare the deployment ticket."
                ],
                "chat_messages": [{
                    "sender": "Alice",
                    "text": "@CTOX wie ist der Status?",
                    "timestamp": "2026-04-28T12:00:00Z"
                }]
            }))
            .unwrap(),
        )
        .expect("session json");

        let request = AdapterSyncCommandRequest {
            db_path: &db_path,
            passthrough_args: &[],
            skip_flags: &[],
        };
        let first = sync(&root, &BTreeMap::new(), &request).expect("first sync");
        let second = sync(&root, &BTreeMap::new(), &request).expect("second sync");
        assert_eq!(first["active_sessions"], 1);
        assert_eq!(first["ingested"], 1);
        assert_eq!(second["active_sessions"], 1);
        assert_eq!(second["ingested"], 0);
        assert_eq!(second["skipped_unchanged_sessions"], 1);

        let stdin_contents = std::fs::read_to_string(&stdin_path).expect("stdin contents");
        let commands = stdin_contents.lines().collect::<Vec<_>>();
        assert_eq!(commands.len(), 1);
        assert!(commands[0].contains("send_chat"));
        assert!(commands[0].contains("Echtzeit-Antworten"));

        let updated_session: Value =
            serde_json::from_str(&std::fs::read_to_string(&session_path).unwrap()).unwrap();
        assert_eq!(updated_session["mention_ack_sent"], true);

        let conn = open_channel_db(&db_path).expect("channel db");
        let mention_metadata: String = conn
            .query_row(
                "SELECT metadata_json FROM communication_messages WHERE channel='meeting' AND direction='inbound' AND body_text LIKE '%@CTOX%'",
                [],
                |row| row.get(0),
            )
            .expect("mention metadata");
        let mention_metadata: Value = serde_json::from_str(&mention_metadata).unwrap();
        assert_eq!(mention_metadata["is_mention"], true);
        assert_eq!(mention_metadata["priority"], "urgent");
        assert_eq!(mention_metadata["transcript_chunk_count"], 2);
        assert!(mention_metadata["transcript_snapshot"]
            .as_str()
            .unwrap_or_default()
            .contains("rollout is blocked"));

        let mention_body: String = conn
            .query_row(
                "SELECT body_text FROM communication_messages WHERE channel='meeting' AND direction='inbound' AND body_text LIKE '%@CTOX%'",
                [],
                |row| row.get(0),
            )
            .expect("mention body");
        assert!(mention_body.contains("Live-Transcript bisher"));
        assert!(mention_body.contains("deployment ticket"));

        let ack_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM communication_messages WHERE metadata_json LIKE '%ctox_first_mention_ack%'",
                [],
                |row| row.get(0),
            )
            .expect("ack count");
        assert_eq!(ack_count, 1);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn service_sync_ingests_active_meeting_chat() {
        let root = temp_root("service-sync");
        let sessions_dir = meeting_sessions_dir(&root);
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
        let stdin_path = sessions_dir.join("session-service.commands.jsonl");
        std::fs::write(&stdin_path, "").expect("stdin file");
        std::fs::write(
            sessions_dir.join("session-service.json"),
            serde_json::to_string_pretty(&json!({
                "session_id": "session-service",
                "provider": "zoom",
                "bot_name": "INF Yoda Notetaker",
                "status": "active",
                "stdin_pipe": stdin_path.display().to_string(),
                "chat_messages": [{
                    "sender": "Alice",
                    "text": "@CTOX bitte pruefen",
                    "timestamp": "2026-04-28T12:00:00Z"
                }]
            }))
            .unwrap(),
        )
        .expect("session json");

        let result = service_sync(&root, &BTreeMap::new())
            .expect("service sync")
            .expect("meeting sync result");
        assert_eq!(result["ingested"], 1);
        let idle_result = service_sync(&root, &BTreeMap::new())
            .expect("idle service sync")
            .expect("meeting sync result");
        assert_eq!(idle_result["active_sessions"], 1);
        assert_eq!(idle_result["ingested"], 0);
        assert_eq!(idle_result["skipped_unchanged_sessions"], 1);

        let conn = open_channel_db(&root.join("runtime/ctox.sqlite3")).expect("channel db");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM communication_messages WHERE channel='meeting' AND thread_key='session-service'",
                [],
                |row| row.get(0),
            )
            .expect("message count");
        assert_eq!(count, 2);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn sync_filters_bot_echoes_from_session_json() {
        let root = temp_root("sync-own-filter");
        let sessions_dir = meeting_sessions_dir(&root);
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
        let db_path = root.join("runtime/ctox.sqlite3");
        std::fs::write(
            sessions_dir.join("session-own.json"),
            serde_json::to_string_pretty(&json!({
                "session_id": "session-own",
                "provider": "zoom",
                "bot_name": "INF Yoda Notetaker",
                "status": "active",
                "outbound_chat_texts": ["Ich pruefe das."],
                "chat_messages": [
                    {"sender": "INF Yoda Notetaker", "text": "Ich pruefe das.", "timestamp": "2026-04-28T12:00:00Z"},
                    {"sender": "Participant", "text": "You20:35CNCIch pruefe das.", "timestamp": "2026-04-28T12:00:01Z"}
                ]
            }))
            .unwrap(),
        )
        .expect("session json");

        let request = AdapterSyncCommandRequest {
            db_path: &db_path,
            passthrough_args: &[],
            skip_flags: &[],
        };
        let result = sync(&root, &BTreeMap::new(), &request).expect("sync");
        assert_eq!(result["active_sessions"], 1);
        assert_eq!(result["ingested"], 0);
        let conn = open_channel_db(&db_path).expect("channel db");
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM communication_messages WHERE channel='meeting'",
                [],
                |row| row.get(0),
            )
            .expect("message count");
        assert_eq!(count, 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn send_writes_chat_command_to_session_pipe_and_records_outbound() {
        let root = temp_root("send-chat");
        let sessions_dir = meeting_sessions_dir(&root);
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
        let db_path = root.join("runtime/ctox.sqlite3");
        let stdin_path = sessions_dir.join("session-send.stdin");
        std::fs::write(&stdin_path, "").expect("stdin file");
        std::fs::write(
            sessions_dir.join("session-send.json"),
            serde_json::to_string_pretty(&json!({
                "session_id": "session-send",
                "provider": "zoom",
                "status": "active",
                "stdin_pipe": stdin_path.display().to_string(),
                "chat_messages": []
            }))
            .unwrap(),
        )
        .expect("session json");

        let request = MeetingSendCommandRequest {
            db_path: &db_path,
            session_id: "session-send",
            body: "Ich pruefe das und melde mich hier.",
        };
        let result = send(&root, &BTreeMap::new(), &request).expect("send result");
        assert_eq!(result["status"], "submitted");
        assert_eq!(result["delivery"]["confirmed"], false);

        let stdin_contents = std::fs::read_to_string(&stdin_path).expect("stdin contents");
        let command: Value = serde_json::from_str(stdin_contents.trim()).expect("stdin json");
        assert_eq!(command["action"], "send_chat");
        assert_eq!(command["text"], "Ich pruefe das und melde mich hier.");
        assert_eq!(command["message_key"], result["message_key"]);

        let conn = open_channel_db(&db_path).expect("channel db");
        let outbound: (i64, String) = conn
            .query_row(
                "SELECT COUNT(*), MIN(status) FROM communication_messages WHERE channel='meeting' AND direction='outbound' AND metadata_json LIKE '%ctox_reply%'",
                [],
                |row| Ok((row.get(0)?, row.get(1)?)),
            )
            .expect("outbound status");
        assert_eq!(outbound, (1, "submitted".to_string()));
        drop(conn);

        assert!(confirm_meeting_outbound_message(
            &root,
            "session-send",
            result["message_key"].as_str().unwrap()
        )
        .expect("runner confirmation"));
        let conn = open_channel_db(&db_path).expect("channel db");
        let confirmed_status: String = conn
            .query_row(
                "SELECT status FROM communication_messages WHERE message_key = ?1",
                [result["message_key"].as_str().unwrap()],
                |row| row.get(0),
            )
            .expect("confirmed status");
        assert_eq!(confirmed_status, "sent");
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn send_command_file_write_error_does_not_record_sent() {
        let root = temp_root("send-chat-write-error");
        let sessions_dir = meeting_sessions_dir(&root);
        std::fs::create_dir_all(&sessions_dir).expect("sessions dir");
        let db_path = root.join("runtime/ctox.sqlite3");
        let _conn = open_channel_db(&db_path).expect("channel db");
        let missing_command_path = sessions_dir.join("missing/session.commands.jsonl");
        std::fs::write(
            sessions_dir.join("session-send-error.json"),
            serde_json::to_string_pretty(&json!({
                "session_id": "session-send-error",
                "provider": "zoom",
                "status": "active",
                "stdin_pipe": missing_command_path.display().to_string(),
                "chat_messages": []
            }))
            .unwrap(),
        )
        .expect("session json");

        let request = MeetingSendCommandRequest {
            db_path: &db_path,
            session_id: "session-send-error",
            body: "Diese Nachricht darf nicht als gesendet gelten.",
        };
        let err = send(&root, &BTreeMap::new(), &request).expect_err("write must fail");
        assert!(err
            .to_string()
            .contains("failed to open meeting runner command file"));

        let conn = open_channel_db(&db_path).expect("channel db");
        let sent_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM communication_messages WHERE channel='meeting' AND direction='outbound' AND status='sent'",
                [],
                |row| row.get(0),
            )
            .expect("sent count");
        let outbound_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM communication_messages WHERE channel='meeting' AND direction='outbound'",
                [],
                |row| row.get(0),
            )
            .expect("outbound count");
        assert_eq!(sent_count, 0);
        assert_eq!(outbound_count, 0);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn simulate_meeting_runs_offline_post_meeting_pipeline() {
        let root = temp_root("simulate");
        let fixture = root.join("fixture.wav");
        std::fs::write(&fixture, vec![0_u8; 4096]).expect("fixture audio");
        let args = vec![
            "--provider".to_string(),
            "zoom".to_string(),
            "--audio".to_string(),
            fixture.display().to_string(),
            "--transcript".to_string(),
            "A participant agreed to create a rollout ticket.".to_string(),
            "--chat".to_string(),
            "Alice:@CTOX bitte als Ticket aufnehmen".to_string(),
        ];

        let result = simulate_meeting_session(&root, &args).expect("simulate");
        assert_eq!(result["ok"], true);
        assert_eq!(result["provider"], "zoom");
        assert_eq!(result["transcript_chunks"], 1);
        assert_eq!(result["chat_messages"], 1);
        assert_eq!(
            result["recording_artifacts"].as_array().map(Vec::len),
            Some(1)
        );
        assert_eq!(result["finalization"]["action"], "ingested");
        let session_id = result["session_id"].as_str().expect("session id");
        let transcript = load_meeting_transcript(&root, session_id).expect("transcript");
        assert!(transcript["transcript"]
            .as_str()
            .unwrap_or_default()
            .contains("rollout ticket"));
        assert!(transcript["chatlog"]
            .as_str()
            .unwrap_or_default()
            .contains("@CTOX"));
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn finalize_meeting_writes_artifact_manifest_and_default_ticket() {
        let root = temp_root("finalize");
        let config = MeetingSessionConfig {
            root: root.clone(),
            meeting_url: "https://meet.google.com/abc-defg-hij".to_string(),
            provider: MeetingProvider::GoogleMeet,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 60,
            audio_chunk_seconds: 30,
            stt_model: String::new(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let mut session = MeetingSession::new(&config);
        session.session_id = "meeting-google-finalize-test".to_string();
        session.status = MeetingSessionStatus::Ended;
        session.ended_at = Some("2026-04-28T12:30:00Z".to_string());
        session
            .transcript_chunks
            .push("A participant agreed to create a deployment ticket.".to_string());
        session.chat_messages.push(ChatMessage {
            sender: "Alice".to_string(),
            text: "Bitte als Ticket aufnehmen.".to_string(),
            timestamp: "2026-04-28T12:10:00Z".to_string(),
        });
        let artifact_dir =
            meeting_sessions_dir(&root).join(format!("{}-audio", session.session_id));
        std::fs::create_dir_all(&artifact_dir).expect("artifact dir");
        std::fs::write(artifact_dir.join("chunk-001.webm"), b"audio").expect("audio artifact");
        std::fs::write(artifact_dir.join("screen-001.mp4"), b"screen").expect("screen artifact");

        let result = finalize_meeting(&root, &session, &config).expect("finalize");
        assert_eq!(result["action"], "ingested");
        assert_eq!(result["recording_artifact_count"], 2);
        assert!(result["post_meeting_ticket_id"].as_str().is_some());

        let transcript_path =
            meeting_sessions_dir(&root).join(format!("{}-transcript.txt", session.session_id));
        let chatlog_path =
            meeting_sessions_dir(&root).join(format!("{}-chatlog.txt", session.session_id));
        let manifest_path =
            meeting_sessions_dir(&root).join(format!("{}-artifacts.json", session.session_id));
        assert!(transcript_path.exists());
        assert!(chatlog_path.exists());
        assert!(manifest_path.exists());
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(manifest_path).unwrap()).unwrap();
        assert_eq!(
            manifest["recording_artifacts"].as_array().map(Vec::len),
            Some(2)
        );
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn detect_cancellation_and_update() {
        assert!(is_meeting_cancellation(
            "Meeting Canceled: Sprint Review",
            ""
        ));
        assert!(is_meeting_cancellation(
            "",
            "BEGIN:VCALENDAR\nMETHOD:CANCEL\nEND:VCALENDAR"
        ));
        assert!(is_meeting_cancellation("Abgesagt: Weekly Standup", ""));
        assert!(!is_meeting_cancellation("Meeting: Sprint Review", ""));

        assert!(is_meeting_update("Updated: Sprint Review", ""));
        assert!(is_meeting_update("Verschoben: Weekly Standup", ""));
        assert!(!is_meeting_update("Meeting: Sprint Review", ""));
    }

    #[test]
    fn cron_for_meeting_time_produces_valid_expression() {
        assert_eq!(
            cron_for_meeting_time("2026-04-15T14:30:00Z"),
            Some("30 14 15 4 *".to_string())
        );
        assert_eq!(
            cron_for_meeting_time("2026-12-01T09:00:00Z"),
            Some("0 9 1 12 *".to_string())
        );
        assert_eq!(cron_for_meeting_time("bad"), None);
    }

    #[test]
    fn timeout_and_inactivity_detection_present_in_script() {
        let config = MeetingSessionConfig {
            root: PathBuf::from("/tmp"),
            meeting_url: "https://meet.google.com/abc".to_string(),
            provider: MeetingProvider::GoogleMeet,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 60,
            audio_chunk_seconds: 30,
            stt_model: String::new(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let script = build_meeting_runner_script_with_timeout(&config).unwrap();
        // Verify transplanted reference detection logic is present
        assert!(
            script.contains("detectLoneParticipant"),
            "missing participant detection"
        );
        assert!(
            script.contains("data-avatar-count"),
            "missing Google Meet badge detection"
        );
        assert!(
            script.contains("detectSilence"),
            "missing silence detection"
        );
        assert!(
            script.contains("AudioContext"),
            "missing AudioContext for silence"
        );
        assert!(
            script.contains("ctoxMeetingEnd"),
            "missing meeting end callback"
        );
        assert!(
            script.contains("verifyJoinedUi"),
            "missing join verification gate"
        );
        assert!(script.contains("join_failed"), "missing join failure event");
        assert!(
            script.contains("CTOX_MEETING_COMMAND_FILE"),
            "missing command file bridge"
        );
        assert!(
            script.contains("commandFilePollInterval"),
            "missing command file poller"
        );
        assert!(
            script.contains("--in-process-gpu"),
            "missing chromium in-process GPU flag"
        );
        assert!(
            script.contains("installChatObservers"),
            "missing chat mutation observer"
        );
        assert!(
            script.contains("transcriptPollInterval"),
            "missing live caption transcript observer"
        );
        assert!(
            script.contains("speakerPollInterval"),
            "missing active speaker observer"
        );
        assert!(
            script.contains("platform_active_speaker"),
            "missing platform active speaker source"
        );
        assert!(
            script.contains("platform_caption"),
            "missing platform caption source"
        );

        // Verify Teams uses ffmpeg path
        let teams_config = MeetingSessionConfig {
            provider: MeetingProvider::MicrosoftTeams,
            ..config.clone()
        };
        let teams_script = build_meeting_runner_script_with_timeout(&teams_config).unwrap();
        assert!(
            teams_script.contains("ffmpeg"),
            "Teams should use ffmpeg recording"
        );
        assert!(
            teams_script.contains("virtual_output.monitor"),
            "Teams should use PulseAudio"
        );
        assert!(
            teams_script.contains("--kiosk"),
            "Teams should use kiosk mode"
        );
        assert!(
            teams_script.contains("warmUpTeamsMediaDevices"),
            "Teams should warm up media devices"
        );
        assert!(
            !teams_script.contains("await enableTeamsLiveCaptions"),
            "Teams must not enable Teams captions; Microsoft meetings use Mistral realtime STT only"
        );
        assert!(
            teams_script.contains("stdoutClosed"),
            "meeting runner should tolerate stdout EPIPE after host shutdown"
        );
        assert!(
            teams_script.contains(r#"type: "recording_artifact""#),
            "Teams should preserve the full ffmpeg recording as an artifact"
        );
        assert!(
            teams_script.contains("mistral_realtime_stt.py"),
            "Teams should write the realtime STT helper"
        );
        assert!(
            teams_script.contains("client.audio.realtime.transcribe_stream"),
            "Teams live transcript must use Mistral realtime streaming"
        );
        assert!(
            teams_script.contains("voxtral-mini-transcribe-realtime-2602"),
            "Teams should default to the Voxtral realtime model"
        );
        assert!(
            teams_script.contains("AudioFormat(encoding=\"pcm_s16le\", sample_rate=16000)"),
            "Teams should stream raw 16 kHz PCM into realtime STT"
        );
        assert!(
            teams_script.contains("target_streaming_delay_ms=delay_ms")
                && teams_script.contains("\"1800\"")
                && teams_script.contains("\"8192\""),
            "Teams realtime STT should use a low-latency streaming configuration"
        );
        assert!(
            teams_script.contains("__ctoxTranscriptOverlayLive")
                && teams_script.contains("__ctoxTranscriptOverlayCommit")
                && teams_script.contains("setTimeout(flushRealtimeBuffer, 2800)"),
            "Teams realtime STT must render streaming deltas as a live line before committing transcript segments"
        );
        assert!(
            !teams_script.contains("teams-audio-chunks"),
            "Teams live transcript must not use file chunk directories"
        );
        assert!(
            !teams_script.contains("audioSegmenter"),
            "Teams live transcript must not use the old batch segmenter"
        );
        assert!(
            !teams_script.contains("Live-ish Teams STT"),
            "Teams should not present delayed batch STT as live transcript"
        );
    }

    #[test]
    fn runner_scripts_keep_provider_recording_paths() {
        let google_config = MeetingSessionConfig {
            root: PathBuf::from("/tmp"),
            meeting_url: "https://meet.google.com/abc".to_string(),
            provider: MeetingProvider::GoogleMeet,
            bot_name: "INF Yoda Notetaker".to_string(),
            max_duration_minutes: 60,
            audio_chunk_seconds: 30,
            stt_model: String::new(),
            realtime_stt_model: "voxtral-mini-transcribe-realtime-2602".to_string(),
            mistral_api_key: None,
        };
        let google_script = build_meeting_runner_script(&google_config).unwrap();
        assert!(google_script.contains("getDisplayMedia"));
        assert!(google_script.contains("ctoxAudioChunk"));
        assert!(google_script.contains("video: true"));

        let zoom_script = build_meeting_runner_script(&MeetingSessionConfig {
            provider: MeetingProvider::Zoom,
            meeting_url: "https://zoom.us/j/123456".to_string(),
            ..google_config.clone()
        })
        .unwrap();
        assert!(zoom_script.contains("getDisplayMedia"));
        assert!(zoom_script.contains("ctoxAudioChunk"));
        assert!(zoom_script.contains("buildZoomWebClientUrl"));
        assert!(zoom_script.contains("button.preview-join-button"));
        assert!(zoom_script.contains("prepareZoomAudio"));
        assert!(zoom_script.contains("startZoomRemovalMonitor"));

        let teams_script = build_meeting_runner_script(&MeetingSessionConfig {
            provider: MeetingProvider::MicrosoftTeams,
            meeting_url: "https://teams.microsoft.com/l/meetup-join/abc".to_string(),
            ..google_config
        })
        .unwrap();
        assert!(teams_script.contains("ffmpeg"));
        assert!(teams_script.contains("recording_artifact"));
        assert!(teams_script.contains(r#"extension: "mp4""#));
        assert!(teams_script.contains("x11grab"));
    }
}

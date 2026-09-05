//! Lossy delivery, durable sources: no cockpit write participates in admission or finalization.
#[cfg(test)]
#[path = "harness_cockpit_projection_tests.rs"]
mod tests;
use super::{queue_pause, queue_retention};
use crate::business_os::store::{self, BusinessProjectionWriter as NativeProjectionWriter};
use anyhow::Result;
use chrono::{DateTime, Utc};
use rusqlite::{params, Connection, OptionalExtension};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

#[derive(Clone, Default, serde::Deserialize)]
#[serde(default)]
pub(crate) struct WorkerSnapshot {
    pub service_running: bool,
    pub busy: bool,
    pub worker_active_count: usize,
    pub worker_phase: Option<String>,
    pub active_task_ids: Vec<String>,
    pub last_error: Option<String>,
    pub boot_id: String,
}

/// Successful payloads only: failed RxDB delivery is retried, and restart replays
/// durable sources. Per-root cache entries are removed with their tombstones.
struct BusinessProjectionWriter {
    inner: NativeProjectionWriter,
    payloads: BTreeMap<(String, String), Value>,
}
impl BusinessProjectionWriter {
    fn open(root: &Path) -> Result<Self> {
        store::open_store(root)?.execute_batch("CREATE INDEX IF NOT EXISTS idx_cockpit_live_records ON business_records(collection,record_id) WHERE deleted=0;
            CREATE INDEX IF NOT EXISTS idx_cockpit_event_task_time ON business_records(json_extract(payload_json,'$.task_id'),json_extract(payload_json,'$.created_at_ms') DESC,record_id DESC) WHERE collection='ctox_harness_events' AND deleted=0;
            CREATE INDEX IF NOT EXISTS idx_cockpit_run_finished ON business_records(json_extract(payload_json,'$.finished_at_ms') DESC,record_id DESC) WHERE collection='ctox_runs' AND deleted=0;")?;
        Ok(Self {
            inner: NativeProjectionWriter::open(root)?,
            payloads: BTreeMap::new(),
        })
    }
    fn upsert_source_projection(
        &mut self,
        collection: &str,
        id: &str,
        source_ms: i64,
        mut payload: Value,
    ) -> Result<()> {
        let key = (collection.to_string(), id.to_string());
        let mut comparable = payload.clone();
        comparable
            .as_object_mut()
            .map(|object| object.remove("updated_at_ms"));
        if self.payloads.get(&key) == Some(&comparable) {
            return Ok(());
        }
        let observed = Utc::now().timestamp_millis().max(source_ms);
        payload["updated_at_ms"] = json!(observed);
        self.inner
            .upsert_source_projection(collection, id, observed, payload)?;
        if self.inner.delivered_to_rxdb(collection) {
            self.payloads.insert(key, comparable);
        }
        Ok(())
    }
    fn tombstone_source_projection(&mut self, collection: &str, id: &str, now: i64) -> Result<()> {
        self.inner
            .tombstone_source_projection(collection, id, now)?;
        self.payloads
            .remove(&(collection.to_string(), id.to_string()));
        Ok(())
    }
}

fn persisted_snapshot(root: &Path) -> Result<WorkerSnapshot> {
    let conn = store::open_store(root)?;
    let raw:Option<String>=conn.query_row("SELECT payload_json FROM business_records WHERE collection='ctox_harness_status' AND record_id='harness' AND deleted=0",[],|row|row.get(0)).optional()?;
    Ok(raw
        .map(|raw| serde_json::from_str(&raw))
        .transpose()?
        .unwrap_or_default())
}

struct Pump {
    wake: mpsc::SyncSender<(PathBuf, u8)>,
    latest: Arc<Mutex<BTreeMap<PathBuf, WorkerSnapshot>>>,
}

const STATUS: u8 = 1;
const EVENTS: u8 = 2;
const RUNS: u8 = 4;
const QUEUE: u8 = 8;
const CHAT: u8 = 16;
const ALL: u8 = STATUS | EVENTS | RUNS | QUEUE | CHAT;

fn pump() -> Option<&'static Pump> {
    static PUMP: OnceLock<Option<Pump>> = OnceLock::new();
    PUMP.get_or_init(|| {
        let (wake, receive) = mpsc::sync_channel::<(PathBuf, u8)>(128);
        let latest = Arc::new(Mutex::new(BTreeMap::<PathBuf, WorkerSnapshot>::new()));
        let snapshots = latest.clone();
        let worker = std::thread::Builder::new()
            .name("cockpit-projections".into())
            .spawn(move || {
                let mut roots = BTreeSet::<PathBuf>::new();
                let mut writers = BTreeMap::<PathBuf, BusinessProjectionWriter>::new();
                let mut next_sweep = Instant::now() + Duration::from_secs(60);
                loop {
                    let mut dirty = BTreeMap::<PathBuf, u8>::new();
                    match receive.recv_timeout(next_sweep.saturating_duration_since(Instant::now()))
                    {
                        Ok((root, flags)) => {
                            dirty.insert(root, flags);
                        }
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                    for (root, flags) in receive.try_iter() {
                        *dirty.entry(root).or_default() |= flags;
                    }
                    if Instant::now() >= next_sweep {
                        for root in &roots {
                            dirty.insert(root.clone(), ALL);
                        }
                        for root in snapshots.lock().unwrap_or_else(|e| e.into_inner()).keys() {
                            dirty.insert(root.clone(), ALL);
                        }
                        next_sweep = Instant::now() + Duration::from_secs(60);
                    }
                    for (root, mut flags) in dirty {
                        if !crate::paths::core_db(&root).is_file() {
                            continue;
                        }
                        if roots.insert(root.clone()) {
                            flags = ALL;
                        }
                        let snapshot = snapshots
                            .lock()
                            .unwrap_or_else(|e| e.into_inner())
                            .get(&root)
                            .cloned();
                        let outcome = (|| -> Result<()> {
                            let snapshot = snapshot
                                .map(Ok)
                                .unwrap_or_else(|| persisted_snapshot(&root))?;
                            if !writers.contains_key(&root) {
                                writers
                                    .insert(root.clone(), BusinessProjectionWriter::open(&root)?);
                            }
                            refresh_selected(
                                &root,
                                &snapshot,
                                writers.get_mut(&root).expect("inserted writer"),
                                flags,
                            )
                        })();
                        if let Err(error) = outcome {
                            eprintln!(
                                "[ctox cockpit] projection deferred for {}: {error:#}",
                                root.display()
                            );
                        }
                    }
                    let removed = roots
                        .iter()
                        .filter(|root| !crate::paths::core_db(root).is_file())
                        .cloned()
                        .collect::<BTreeSet<_>>();
                    roots.retain(|root| !removed.contains(root));
                    writers.retain(|root, _| !removed.contains(root));
                    snapshots
                        .lock()
                        .unwrap_or_else(|e| e.into_inner())
                        .retain(|root, _| !removed.contains(root));
                }
            });
        match worker {
            Ok(_) => Some(Pump { wake, latest }),
            Err(error) => {
                eprintln!("[ctox cockpit] projection worker unavailable: {error}");
                None
            }
        }
    })
    .as_ref()
}

fn wake(root: &Path, flags: u8) {
    if let Some(pump) = pump() {
        if pump.wake.try_send((root.to_path_buf(), flags)).is_err() {
            static DROPPED: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
            let dropped = DROPPED.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if dropped.is_power_of_two() {
                eprintln!("[ctox cockpit] coalesced wake dropped ({dropped}); durable sources replay on maintenance");
            }
        }
    }
}
pub(crate) fn schedule_flow_refresh(root: &Path, kind: &str) {
    let chat = matches!(
        kind,
        "worker.plan_updated" | "worker.turn_started" | "cockpit.review"
    );
    wake(root, EVENTS | if chat { CHAT } else { 0 });
}
pub(crate) fn schedule_runs_refresh(root: &Path) {
    wake(root, RUNS);
}

pub(crate) fn publish_service_stopped(root: &Path, boot_id: String) {
    let snapshot = WorkerSnapshot {
        boot_id,
        last_error: persisted_snapshot(root).ok().and_then(|s| s.last_error),
        ..Default::default()
    };
    publish_worker_snapshot(root, snapshot.clone());
    // Graceful process exit cannot rely on a detached worker being scheduled before termination.
    let outcome = (|| -> Result<()> {
        let conn = core(root)?;
        if has_table(&conn, "communication_routing_state")? {
            project_status(
                root,
                &conn,
                &mut BusinessProjectionWriter::open(root)?,
                &snapshot,
            )?;
        }
        Ok(())
    })();
    if let Err(error) = outcome {
        eprintln!("[ctox cockpit] shutdown status write failed: {error:#}");
    }
}

pub(crate) fn schedule_refresh(root: &Path) {
    wake(root, STATUS | QUEUE | CHAT);
}

/// Only a short in-memory update and a nonblocking wake, including when called under SharedState.
pub(crate) fn publish_worker_snapshot(root: &Path, snapshot: WorkerSnapshot) {
    if let Some(pump) = pump() {
        pump.latest
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .insert(root.to_path_buf(), snapshot);
        wake(root, STATUS);
    }
}

pub(crate) fn refresh_after_finalization(db_path: &Path) {
    if let Some(root) = db_path.parent().and_then(Path::parent) {
        wake(root, STATUS | RUNS | QUEUE | CHAT);
    }
}

fn millis(value: &str) -> i64 {
    // LCM finalizations use epoch-millisecond strings; flow/routing use RFC3339.
    if let Ok(value) = value.parse::<i64>() {
        return value;
    }
    DateTime::parse_from_rfc3339(value)
        .map(|date| date.timestamp_millis())
        .unwrap_or(0)
}

fn has_table(conn: &Connection, table: &str) -> Result<bool> {
    Ok(conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE type='table' AND name=?1)",
        [table],
        |row| row.get(0),
    )?)
}

fn core(root: &Path) -> Result<Connection> {
    let conn = Connection::open(crate::paths::core_db(root))?;
    conn.busy_timeout(Duration::from_millis(100))?;
    // Indexes are additive. The flow ledger remains the authoritative evidence.
    if has_table(&conn, "ctox_harness_flow_events")? {
        conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_cockpit_flow_attempt ON ctox_harness_flow_events(json_extract(metadata_json,'$.attempt_id'),created_at);
            CREATE INDEX IF NOT EXISTS idx_cockpit_flow_task_time ON ctox_harness_flow_events(message_key,created_at DESC);")?;
    }
    if has_table(&conn, "worker_attempt_finalizations")? {
        conn.execute_batch("CREATE INDEX IF NOT EXISTS idx_cockpit_finalized_time ON worker_attempt_finalizations(COALESCE(terminal_at,updated_at) DESC,attempt_id DESC) WHERE status!='finalizing';")?;
    }
    if has_table(&conn, "api_model_cost_events")? {
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_cockpit_cost_turn ON api_model_cost_events(turn_id);",
        )?;
    }
    Ok(conn)
}

fn refresh_selected(
    root: &Path,
    snapshot: &WorkerSnapshot,
    writer: &mut BusinessProjectionWriter,
    flags: u8,
) -> Result<()> {
    let conn = core(root)?;
    if !has_table(&conn, "communication_routing_state")? {
        return Ok(());
    }
    if flags & STATUS != 0 {
        project_status(root, &conn, writer, snapshot)?;
    }
    if flags & EVENTS != 0 {
        project_events(root, &conn, writer)?;
    }
    if flags & RUNS != 0 {
        project_runs(root, &conn, writer)?;
    }
    if flags & CHAT != 0 && has_table(&conn, "ctox_harness_flow_events")? {
        super::chat::project(root, &conn)?;
    }
    let mut collections = Vec::new();
    if flags & QUEUE != 0 {
        collections.push("ctox_queue_tasks");
    }
    if flags & EVENTS != 0 {
        collections.push("ctox_harness_events");
    }
    if flags & RUNS != 0 {
        collections.push("ctox_runs");
    }
    retain_selected(
        root,
        &conn,
        writer,
        Utc::now().timestamp_millis(),
        &collections,
    )?;
    Ok(())
}

fn project_status(
    root: &Path,
    conn: &Connection,
    writer: &mut BusinessProjectionWriter,
    snapshot: &WorkerSnapshot,
) -> Result<()> {
    let mut counts = BTreeMap::new();
    let mut snapshot = snapshot.clone();
    if crate::service::cockpit_service_running(root) == Some(false) {
        snapshot.service_running = false;
        snapshot.busy = false;
        snapshot.worker_active_count = 0;
        snapshot.worker_phase = None;
        snapshot.active_task_ids.clear();
    }
    let mut statement = conn.prepare(
        "SELECT r.route_status,COUNT(*) FROM communication_routing_state r JOIN communication_messages m ON m.message_key=r.message_key WHERE m.channel='queue' AND m.direction='inbound' GROUP BY r.route_status",
    )?;
    for row in statement.query_map([], |row| {
        Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
    })? {
        let (status, count) = row?;
        counts.insert(status, count);
    }
    let count = |status: &str| counts.get(status).copied().unwrap_or(0);
    let failed_recent: i64 = conn.query_row("SELECT COUNT(*) FROM communication_routing_state r JOIN communication_messages m ON m.message_key=r.message_key WHERE m.channel='queue' AND m.direction='inbound' AND r.route_status='failed' AND julianday(r.updated_at)>=julianday('now','-1 day')",[],|row|row.get(0))?;
    let review_count = if has_table(conn, "business_command_aggregates")?
        && has_table(conn, "business_command_task_links")?
    {
        conn.query_row("SELECT COUNT(*) FROM communication_routing_state r JOIN communication_messages m ON m.message_key=r.message_key WHERE m.channel='queue' AND m.direction='inbound' AND (r.route_status='review_rework' OR EXISTS(SELECT 1 FROM business_command_task_links l JOIN business_command_aggregates a ON a.command_id=l.command_id WHERE l.task_id=r.message_key AND a.execution_phase='awaiting_review'))",[],|row|row.get::<_,i64>(0))?
    } else {
        count("review_rework")
    };
    let pause = queue_pause(root)?;
    let now = Utc::now().timestamp_millis();
    let capacity = crate::service::configure_queue_worker_capacity(root, None)?;
    let threshold = crate::service::queue_pressure_threshold();
    writer.upsert_source_projection("ctox_harness_status","harness",now,json!({
        "id":"harness", "service_running":snapshot.service_running, "busy":snapshot.busy,
        "paused":pause.paused,"pause_reason":pause.reason,
        "worker_active_count":snapshot.worker_active_count,"worker_phase":snapshot.worker_phase,
        "worker_capacity":capacity["max_workers"],"pending_count":count("pending"),"leased_count":count("leased"),
        "blocked_count":count("blocked"),"review_count":review_count,"failed_recent_count":failed_recent,
        "pressure_active":count("pending") >= threshold as i64,"pressure_threshold":threshold,
        "work_hours":crate::service::working_hours::snapshot(root),"active_task_ids":snapshot.active_task_ids,
        "active_crew_member_id":null,"last_error":snapshot.last_error,"boot_id":snapshot.boot_id,"updated_at_ms":now
    }))?;
    Ok(())
}

fn event_kind(kind: &str) -> Option<&'static str> {
    Some(match kind {
        "worker.tool_started" => "tool_started",
        "worker.tool_completed" => "tool_completed",
        "worker.thinking_started" | "worker.thinking" => "thinking",
        "worker.plan_updated" => "plan_updated",
        "worker.token_usage" => "token_usage",
        "worker.turn_completed" => "turn_completed",
        "worker.phase" | "worker.turn_started" => "phase",
        "crew.selected" | "crew_selected" => "crew_selected",
        _ => return None,
    })
}

fn project_events(
    _root: &Path,
    conn: &Connection,
    writer: &mut BusinessProjectionWriter,
) -> Result<()> {
    if !has_table(conn, "ctox_harness_flow_events")? {
        return Ok(());
    }
    // A short turn may finish before this lossy worker wakes. Eligibility was recorded
    // at emission; replay recent terminal tasks too so their last events are not lost.
    let mut tasks = conn.prepare("SELECT message_key FROM communication_routing_state WHERE route_status NOT IN ('handled','failed','cancelled') OR julianday(updated_at)>=julianday('now','-1 day')")?;
    let tasks = tasks
        .query_map([], |row| row.get::<_, String>(0))?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for task in tasks {
        let mut statement=conn.prepare("SELECT event_id,event_kind,title,attempt_index,metadata_json,created_at FROM ctox_harness_flow_events
            WHERE message_key=?1 AND COALESCE(json_extract(metadata_json,'$.cockpit_eligible'),1)=1 AND event_kind IN ('worker.turn_started','worker.tool_started','worker.tool_completed','worker.thinking_started','worker.thinking','worker.plan_updated','worker.token_usage','worker.turn_completed','worker.phase','crew.selected','crew_selected')
            ORDER BY created_at DESC,event_id DESC LIMIT 200")?;
        let events = statement
            .query_map([&task], |r| {
                Ok((
                    r.get::<_, String>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                    r.get::<_, Option<i64>>(3)?,
                    r.get::<_, String>(4)?,
                    r.get::<_, String>(5)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        for (id, kind, title, attempt, metadata, created_at) in events {
            let metadata: Value = serde_json::from_str(&metadata)?;
            let created_at_ms = millis(&created_at);
            let usage = metadata.get("usage").unwrap_or(&Value::Null);
            // Tool arguments/results and raw reasoning are deliberately absent from this projection.
            writer.upsert_source_projection("ctox_harness_events",&id,created_at_ms,json!({
                "id":id,"task_id":task,"command_id":metadata.get("command_id"),"attempt":attempt.or_else(||metadata.get("attempt").and_then(Value::as_i64)),
                "kind":event_kind(&kind),"title":title,"tool_type":metadata.pointer("/tool/type"),"tool_name":metadata.pointer("/tool/name"),"call_id":metadata.pointer("/tool/call_id"),
                "success":metadata.pointer("/tool/success"),"usage":{"input":usage.get("input_tokens"),"output":usage.get("output_tokens"),"reasoning":usage.get("reasoning_output_tokens"),"total":usage.get("total_tokens")},
                "runtime_seconds":metadata.pointer("/runtime/seconds"),"step_position":metadata.get("step_position"),
                "created_at_ms":created_at_ms,"updated_at_ms":created_at_ms
            }))?;
        }
    }
    Ok(())
}

fn project_runs(
    root: &Path,
    conn: &Connection,
    writer: &mut BusinessProjectionWriter,
) -> Result<()> {
    if !has_table(conn, "worker_attempt_finalizations")?
        || !has_table(conn, "ctox_harness_flow_events")?
    {
        return Ok(());
    }
    let mut statement=conn.prepare("WITH selected AS (
        SELECT attempt_id FROM (SELECT attempt_id FROM worker_attempt_finalizations WHERE status!='finalizing' ORDER BY COALESCE(terminal_at,updated_at) DESC,attempt_id DESC LIMIT 500)
        UNION SELECT DISTINCT json_extract(e.metadata_json,'$.attempt_id') FROM communication_routing_state r JOIN ctox_harness_flow_events e ON e.message_key=r.message_key WHERE r.route_status NOT IN ('handled','failed','cancelled') AND json_extract(e.metadata_json,'$.attempt_id') IS NOT NULL
        )
        SELECT f.attempt_id,(SELECT e.work_id FROM ctox_harness_flow_events e WHERE json_extract(e.metadata_json,'$.attempt_id')=f.attempt_id AND e.work_id IS NOT NULL ORDER BY e.created_at LIMIT 1),f.status,f.agent_outcome,f.created_at,COALESCE(f.terminal_at,f.updated_at),f.error_text,f.resumable,
        (SELECT e.message_key FROM ctox_harness_flow_events e WHERE json_extract(e.metadata_json,'$.attempt_id')=f.attempt_id AND e.message_key IS NOT NULL ORDER BY e.created_at LIMIT 1)
        FROM selected s JOIN worker_attempt_finalizations f ON f.attempt_id=s.attempt_id WHERE f.status!='finalizing'
        ORDER BY f.updated_at LIMIT (SELECT COUNT(*) FROM selected)")?;
    let rows = statement
        .query_map([], |r| {
            Ok((
                r.get::<_, String>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, String>(2)?,
                r.get::<_, String>(3)?,
                r.get::<_, String>(4)?,
                r.get::<_, String>(5)?,
                r.get::<_, Option<String>>(6)?,
                r.get::<_, bool>(7)?,
                r.get::<_, Option<String>>(8)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?;
    for (id, work, status, outcome, created, finished, error, resumable, task) in rows {
        let (started,command_id):(Option<String>,Option<String>)=conn.query_row("SELECT MIN(created_at),MAX(json_extract(metadata_json,'$.command_id')) FROM ctox_harness_flow_events WHERE json_extract(metadata_json,'$.attempt_id')=?1",[&id],|r|Ok((r.get(0)?,r.get(1)?)))?;
        let started_ms = millis(started.as_deref().unwrap_or(&created));
        let finished_ms = millis(&finished);
        let mut metrics = run_metrics(root, conn, &id)?;
        metrics["elapsed_ms"] = json!((finished_ms - started_ms).max(0));
        let review:Option<String>=conn.query_row("SELECT json_extract(metadata_json,'$.review') FROM ctox_harness_flow_events WHERE json_extract(metadata_json,'$.attempt_id')=?1 AND json_extract(metadata_json,'$.review') IS NOT NULL ORDER BY created_at DESC LIMIT 1",[&id],|r|r.get(0)).optional()?;
        let review = review
            .and_then(|r| serde_json::from_str::<Value>(&r).ok())
            .unwrap_or(json!({"disposition":null,"hold_reason":null}));
        writer.upsert_source_projection("ctox_runs",&id,finished_ms,json!({
            "id":id,"attempt_id":id,"task_id":task,"command_id":command_id,"work_id":work,"crew_member_id":null,
            "status":status,"agent_outcome":outcome,"started_at_ms":started_ms,"finished_at_ms":finished_ms,
            "metrics":metrics,"review":review,"error_text":error,"resumable":resumable,"retrospective":null,"updated_at_ms":finished_ms
        }))?;
    }
    Ok(())
}

fn run_metrics(root: &Path, conn: &Connection, attempt: &str) -> Result<Value> {
    let (tools,thinking):(i64,i64)=conn.query_row("SELECT COUNT(DISTINCT CASE WHEN event_kind='worker.tool_started' THEN COALESCE(json_extract(metadata_json,'$.tool.call_id'),event_id) END),COUNT(DISTINCT CASE WHEN event_kind='worker.thinking_started' THEN COALESCE(json_extract(metadata_json,'$.activity.id'),event_id) END) FROM ctox_harness_flow_events WHERE json_extract(metadata_json,'$.attempt_id')=?1",[attempt],|r|Ok((r.get(0)?,r.get(1)?)))?;
    let mut result = json!({"model":null,"provider":null,"input_tokens":null,"output_tokens":null,"reasoning_tokens":null,"cost_usd":null,"tool_calls":tools,"thinking_turns":thinking});
    let _ = root;
    if !has_table(conn, "api_model_cost_events")? {
        return Ok(result);
    }
    let (input,output,reasoning,provider,model):(Option<i64>,Option<i64>,Option<i64>,Option<String>,Option<String>)=conn.query_row("SELECT SUM(input_tokens),SUM(output_tokens),SUM(reasoning_output_tokens),GROUP_CONCAT(DISTINCT provider),GROUP_CONCAT(DISTINCT model) FROM api_model_cost_events WHERE turn_id IN (SELECT DISTINCT json_extract(metadata_json,'$.turn_id') FROM ctox_harness_flow_events WHERE json_extract(metadata_json,'$.attempt_id')=?1)",[attempt],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?)))?;
    result["input_tokens"] = json!(input);
    result["output_tokens"] = json!(output);
    result["reasoning_tokens"] = json!(reasoning);
    result["provider"] = json!(provider);
    result["model"] = json!(model);
    if has_table(conn, "api_model_price_rates")? {
        let (count,priced,cost):(i64,i64,Option<f64>)=conn.query_row("SELECT COUNT(*),COUNT(p.model),SUM(((e.input_tokens-e.cached_input_tokens)*p.input_usd_per_million+e.cached_input_tokens*COALESCE(p.cached_input_usd_per_million,p.input_usd_per_million)+e.output_tokens*p.output_usd_per_million)/1000000.0)
            FROM api_model_cost_events e LEFT JOIN api_model_price_rates p ON p.provider=e.provider AND p.model=e.model AND p.effective_from_day=(SELECT MAX(p2.effective_from_day) FROM api_model_price_rates p2 WHERE p2.provider=e.provider AND p2.model=e.model AND p2.effective_from_day<=e.day)
            WHERE e.turn_id IN (SELECT DISTINCT json_extract(metadata_json,'$.turn_id') FROM ctox_harness_flow_events WHERE json_extract(metadata_json,'$.attempt_id')=?1)",[attempt],|r|Ok((r.get(0)?,r.get(1)?,r.get(2)?)))?;
        if count > 0 && count == priced {
            result["cost_usd"] = json!(cost);
        }
    }
    Ok(result)
}

#[cfg(test)]
fn retain(
    root: &Path,
    core: &Connection,
    writer: &mut BusinessProjectionWriter,
    now: i64,
) -> Result<()> {
    retain_selected(
        root,
        core,
        writer,
        now,
        &["ctox_queue_tasks", "ctox_harness_events", "ctox_runs"],
    )
}

fn retain_selected(
    root: &Path,
    core: &Connection,
    writer: &mut BusinessProjectionWriter,
    now: i64,
    collections: &[&str],
) -> Result<()> {
    let business = store::open_store(root)?;
    business.execute(
        "ATTACH DATABASE ?1 AS cockpit_core",
        [crate::paths::core_db(root).to_string_lossy().as_ref()],
    )?;
    let retention = queue_retention(root)?;
    for &collection in collections {
        let predicate=match collection {
            "ctox_queue_tasks"=>"record_id NOT IN (SELECT message_key FROM cockpit_core.communication_routing_state WHERE route_status IN ('handled','failed','cancelled') ORDER BY updated_at DESC,message_key DESC LIMIT ?2) AND EXISTS(SELECT 1 FROM cockpit_core.communication_routing_state r WHERE r.message_key=record_id AND r.route_status IN ('handled','failed','cancelled'))",
            "ctox_harness_events"=>"json_extract(payload_json,'$.task_id') IN (SELECT message_key FROM cockpit_core.communication_routing_state WHERE route_status IN ('handled','failed','cancelled') AND julianday(updated_at)<julianday(?3/1000.0,'unixepoch','-1 day')) OR record_id IN (SELECT record_id FROM (SELECT record_id,ROW_NUMBER() OVER (PARTITION BY json_extract(payload_json,'$.task_id') ORDER BY json_extract(payload_json,'$.created_at_ms') DESC,record_id DESC) position FROM business_records WHERE collection='ctox_harness_events' AND deleted=0) WHERE position>200)",
            _=>"record_id NOT IN (SELECT record_id FROM business_records WHERE collection='ctox_runs' AND deleted=0 ORDER BY json_extract(payload_json,'$.finished_at_ms') DESC,record_id DESC LIMIT 500) AND NOT EXISTS(SELECT 1 FROM cockpit_core.communication_routing_state r WHERE r.message_key=json_extract(payload_json,'$.task_id') AND r.route_status NOT IN ('handled','failed','cancelled'))",
        };
        let sql=format!("SELECT record_id FROM business_records WHERE collection=?1 AND deleted=0 AND ({predicate}) AND ?2>=0 AND ?3>=0 AND record_id>?4 ORDER BY record_id LIMIT 256");
        let mut cursor = String::new();
        loop {
            let ids = business
                .prepare(&sql)?
                .query_map(params![collection, retention, now, cursor], |r| {
                    r.get::<_, String>(0)
                })?
                .collect::<rusqlite::Result<Vec<_>>>()?;
            if ids.is_empty() {
                break;
            }
            for id in ids {
                cursor = id.clone();
                writer.tombstone_source_projection(collection, &id, now)?;
                // A release/retry can commit while this lossy sweep is writing the mirrors.
                // Re-read after the tombstone: an earlier transition is repaired here; a later
                // transition's normal source callback writes after our tombstone.
                if collection == "ctox_queue_tasks" {
                    let active:bool=core.query_row("SELECT EXISTS(SELECT 1 FROM communication_routing_state WHERE message_key=?1 AND route_status NOT IN ('handled','failed','cancelled'))",[&id],|r|r.get(0))?;
                    if active {
                        if let Some(task) = crate::mission::channels::load_queue_task(root, &id)? {
                            if store::refresh_business_command_queue_task_projection(root, &id)?
                                .is_none()
                            {
                                let payload =
                                    crate::business_os::store_projections::queue_task_payload(
                                        None, &task, None, now,
                                    );
                                writer.upsert_source_projection(collection, &id, now, payload)?;
                            }
                        }
                    }
                }
            }
        }
    }
    let _ = core;
    Ok(())
}

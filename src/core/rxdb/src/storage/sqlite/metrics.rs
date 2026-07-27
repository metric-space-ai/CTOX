//! Runtime metrics for the SQLite storage backend.

use std::collections::{BTreeMap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex as StdMutex, OnceLock};
use std::time::{Duration, Instant};

use serde_json::Value;

macro_rules! atomic_counters {
    ($($name:ident),+ $(,)?) => {
        $(pub(crate) static $name: AtomicU64 = AtomicU64::new(0);)+
    };
}

atomic_counters!(
    SQLITE_BULK_WRITE_CALLS,
    SQLITE_BULK_WRITE_ROWS,
    SQLITE_FIND_DOCUMENTS_BY_ID_CALLS,
    SQLITE_FIND_DOCUMENTS_BY_ID_REQUESTED,
    SQLITE_FIND_DOCUMENTS_BY_ID_RESULTS,
    SQLITE_CHANGED_DOCUMENTS_SINCE_CALLS,
    SQLITE_CHANGED_DOCUMENTS_SINCE_RESULTS,
    SQLITE_QUERY_CALLS,
    SQLITE_QUERY_RESULTS,
    SQLITE_QUERY_FALLBACK_CALLS,
    SQLITE_QUERY_FALLBACK_ROWS_VISITED,
    SQLITE_QUERY_FALLBACK_INDEXED_CANDIDATE_CALLS,
    SQLITE_QUERY_FALLBACK_TOO_BROAD_CALLS,
    SQLITE_COUNT_CALLS,
    SQLITE_COUNT_FALLBACK_QUERY_CALLS,
    SQLITE_QUERY_STREAM_CALLS,
    SQLITE_QUERY_STREAM_RESULTS,
    SQLITE_QUERY_STREAM_UNSUPPORTED_CALLS,
    SQLITE_READ_ONLY_OPEN_CALLS,
    SQLITE_READ_ONLY_OPEN_FAILURES,
    SQLITE_WRITER_FALLBACKS,
    SQLITE_STATEMENTS_EXECUTED,
    SQLITE_STATEMENT_ELAPSED_NS_TOTAL,
    SQLITE_STATEMENT_ELAPSED_NS_MAX,
    SQLITE_STATEMENT_ELAPSED_GE_1MS,
    SQLITE_STATEMENT_ELAPSED_GE_10MS,
    SQLITE_STATEMENT_ELAPSED_GE_100MS,
    SQLITE_STATEMENT_ELAPSED_GE_1000MS,
    SQLITE_WRITE_TRANSACTIONS_STARTED,
    SQLITE_WRITE_TRANSACTIONS_COMMITTED,
    SQLITE_WRITE_TRANSACTIONS_FAILED,
    SQLITE_WRITER_LOCK_ACQUIRE_CALLS,
    SQLITE_WRITER_LOCK_WAIT_NS_TOTAL,
    SQLITE_WRITER_LOCK_WAIT_NS_MAX,
    SQLITE_WRITER_LOCK_WAIT_GE_1MS,
    SQLITE_WRITER_LOCK_WAIT_GE_10MS,
    SQLITE_WRITER_LOCK_WAIT_GE_100MS,
    SQLITE_WRITER_LOCK_WAIT_GE_1000MS,
    SQLITE_WRITER_LOCK_HELD_NS_TOTAL,
    SQLITE_WRITER_LOCK_HELD_NS_MAX,
    SQLITE_WRITER_LOCK_HELD_GE_1MS,
    SQLITE_WRITER_LOCK_HELD_GE_10MS,
    SQLITE_WRITER_LOCK_HELD_GE_100MS,
    SQLITE_WRITER_LOCK_HELD_GE_1000MS,
    SQLITE_EXTERNAL_POLL_DATA_VERSION_READS,
    SQLITE_EXTERNAL_POLL_CHANGED_TABLE_READS,
    SQLITE_EXTERNAL_POLL_CONNECTION_OPENS,
    SQLITE_EXTERNAL_POLL_CONNECTION_OPEN_FAILURES,
    SQLITE_EXTERNAL_POLL_WAKEUPS,
    SQLITE_EXTERNAL_POLL_ACTIVE_WAKEUPS,
    SQLITE_EXTERNAL_POLL_STANDBY_WAKEUPS,
    SQLITE_EXTERNAL_POLL_STANDBY_ENTRIES,
    SQLITE_EXTERNAL_POLL_ACTIVE_RESETS,
    SQLITE_EXTERNAL_POLL_DATA_VERSION_CHANGES,
    SQLITE_EXTERNAL_POLL_DATA_VERSION_READ_FAILURES,
    SQLITE_EXTERNAL_POLL_CHANGED_TABLE_READ_FAILURES,
    SQLITE_EXTERNAL_POLL_CHANGED_TABLE_ROWS,
    SQLITE_EXTERNAL_POLL_CHANGED_TABLE_NOTIFICATIONS,
    SQLITE_EXTERNAL_POLL_LOCAL_HOOK_SUPPRESSED_NOTIFICATIONS,
    SQLITE_EXTERNAL_POLL_DRAIN_CALLS,
    SQLITE_EXTERNAL_POLL_DRAIN_BATCHES,
    SQLITE_EXTERNAL_POLL_DRAIN_EMPTY_BATCHES,
    SQLITE_EXTERNAL_POLL_DRAIN_ROWS_VISITED,
    SQLITE_EXTERNAL_POLL_DRAIN_ROWS_MAX,
    SQLITE_EXTERNAL_POLL_DRAIN_BATCHES_MAX,
    SQLITE_EXTERNAL_POLL_DRAIN_BUDGET_EXHAUSTIONS,
    SQLITE_EXTERNAL_POLL_DRAIN_FAILURES,
);

static SQLITE_QUERY_FALLBACK_BY_COLLECTION: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();
static SQLITE_QUERY_FALLBACK_BY_OPERATOR: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();
static SQLITE_QUERY_FALLBACK_BY_COLLECTION_OPERATOR: OnceLock<
    StdMutex<HashMap<String, HashMap<String, u64>>>,
> = OnceLock::new();
static SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_COLLECTION: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();
static SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_OPERATOR: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();
static SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_COLLECTION_OPERATOR: OnceLock<
    StdMutex<HashMap<String, HashMap<String, u64>>>,
> = OnceLock::new();
static SQLITE_EXTERNAL_POLL_WAKEUPS_BY_DATABASE: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();
static SQLITE_EXTERNAL_POLL_NOTIFICATIONS_BY_TABLE: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();
static SQLITE_EXTERNAL_POLL_LOCAL_HOOK_SUPPRESSIONS_BY_TABLE: OnceLock<
    StdMutex<HashMap<String, u64>>,
> = OnceLock::new();
static SQLITE_EXTERNAL_POLL_DRAIN_ROWS_BY_TABLE: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();
static SQLITE_EXTERNAL_POLL_DRAIN_BATCHES_BY_TABLE: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();
static SQLITE_EXTERNAL_POLL_DRAIN_BUDGET_EXHAUSTIONS_BY_TABLE: OnceLock<
    StdMutex<HashMap<String, u64>>,
> = OnceLock::new();
static SQLITE_EXTERNAL_POLL_DRAIN_FAILURES_BY_TABLE: OnceLock<StdMutex<HashMap<String, u64>>> =
    OnceLock::new();

pub fn sqlite_runtime_counters_snapshot() -> Value {
    let mut out = serde_json::Map::new();
    out.insert(
        "schema".to_string(),
        Value::String("ctox.rxdb.sqlite.runtime_counters.v1".to_string()),
    );
    macro_rules! counter {
        ($name:literal, $value:ident) => {
            out.insert(
                $name.to_string(),
                Value::from($value.load(Ordering::Relaxed)),
            );
        };
    }
    counter!("bulk_write_calls", SQLITE_BULK_WRITE_CALLS);
    counter!("bulk_write_rows", SQLITE_BULK_WRITE_ROWS);
    counter!(
        "find_documents_by_id_calls",
        SQLITE_FIND_DOCUMENTS_BY_ID_CALLS
    );
    counter!(
        "find_documents_by_id_requested",
        SQLITE_FIND_DOCUMENTS_BY_ID_REQUESTED
    );
    counter!(
        "find_documents_by_id_results",
        SQLITE_FIND_DOCUMENTS_BY_ID_RESULTS
    );
    counter!(
        "changed_documents_since_calls",
        SQLITE_CHANGED_DOCUMENTS_SINCE_CALLS
    );
    counter!(
        "changed_documents_since_results",
        SQLITE_CHANGED_DOCUMENTS_SINCE_RESULTS
    );
    counter!("query_calls", SQLITE_QUERY_CALLS);
    counter!("query_results", SQLITE_QUERY_RESULTS);
    counter!("query_fallback_calls", SQLITE_QUERY_FALLBACK_CALLS);
    counter!(
        "query_fallback_rows_visited",
        SQLITE_QUERY_FALLBACK_ROWS_VISITED
    );
    // Every fallback row is decoded before the Rust matcher sees it, so keep the
    // established runtime key without maintaining a duplicate counter family.
    counter!(
        "query_fallback_rows_decoded",
        SQLITE_QUERY_FALLBACK_ROWS_VISITED
    );
    counter!(
        "query_fallback_indexed_candidate_calls",
        SQLITE_QUERY_FALLBACK_INDEXED_CANDIDATE_CALLS
    );
    counter!(
        "query_fallback_too_broad_calls",
        SQLITE_QUERY_FALLBACK_TOO_BROAD_CALLS
    );
    out.insert(
        "query_fallback_by_collection".to_string(),
        snapshot_counter_map(&SQLITE_QUERY_FALLBACK_BY_COLLECTION),
    );
    out.insert(
        "query_fallback_by_operator".to_string(),
        snapshot_counter_map(&SQLITE_QUERY_FALLBACK_BY_OPERATOR),
    );
    out.insert(
        "query_fallback_by_collection_operator".to_string(),
        snapshot_nested_counter_map(&SQLITE_QUERY_FALLBACK_BY_COLLECTION_OPERATOR),
    );
    out.insert(
        "query_fallback_rows_visited_by_collection".to_string(),
        snapshot_counter_map(&SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_COLLECTION),
    );
    out.insert(
        "query_fallback_rows_decoded_by_collection".to_string(),
        snapshot_counter_map(&SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_COLLECTION),
    );
    out.insert(
        "query_fallback_rows_visited_by_operator".to_string(),
        snapshot_counter_map(&SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_OPERATOR),
    );
    out.insert(
        "query_fallback_rows_decoded_by_operator".to_string(),
        snapshot_counter_map(&SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_OPERATOR),
    );
    out.insert(
        "query_fallback_rows_visited_by_collection_operator".to_string(),
        snapshot_nested_counter_map(&SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_COLLECTION_OPERATOR),
    );
    out.insert(
        "query_fallback_rows_decoded_by_collection_operator".to_string(),
        snapshot_nested_counter_map(&SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_COLLECTION_OPERATOR),
    );
    counter!("count_calls", SQLITE_COUNT_CALLS);
    counter!(
        "count_fallback_query_calls",
        SQLITE_COUNT_FALLBACK_QUERY_CALLS
    );
    counter!("query_stream_calls", SQLITE_QUERY_STREAM_CALLS);
    counter!("query_stream_results", SQLITE_QUERY_STREAM_RESULTS);
    counter!(
        "query_stream_unsupported_calls",
        SQLITE_QUERY_STREAM_UNSUPPORTED_CALLS
    );
    counter!("read_only_open_calls", SQLITE_READ_ONLY_OPEN_CALLS);
    counter!("read_only_open_failures", SQLITE_READ_ONLY_OPEN_FAILURES);
    counter!("writer_fallbacks", SQLITE_WRITER_FALLBACKS);
    counter!("statements_executed", SQLITE_STATEMENTS_EXECUTED);
    counter!(
        "statement_elapsed_ns_total",
        SQLITE_STATEMENT_ELAPSED_NS_TOTAL
    );
    counter!("statement_elapsed_ns_max", SQLITE_STATEMENT_ELAPSED_NS_MAX);
    counter!("statement_elapsed_ge_1ms", SQLITE_STATEMENT_ELAPSED_GE_1MS);
    counter!(
        "statement_elapsed_ge_10ms",
        SQLITE_STATEMENT_ELAPSED_GE_10MS
    );
    counter!(
        "statement_elapsed_ge_100ms",
        SQLITE_STATEMENT_ELAPSED_GE_100MS
    );
    counter!(
        "statement_elapsed_ge_1000ms",
        SQLITE_STATEMENT_ELAPSED_GE_1000MS
    );
    counter!(
        "write_transactions_started",
        SQLITE_WRITE_TRANSACTIONS_STARTED
    );
    counter!(
        "write_transactions_committed",
        SQLITE_WRITE_TRANSACTIONS_COMMITTED
    );
    counter!(
        "write_transactions_failed",
        SQLITE_WRITE_TRANSACTIONS_FAILED
    );
    counter!(
        "writer_lock_acquire_calls",
        SQLITE_WRITER_LOCK_ACQUIRE_CALLS
    );
    counter!(
        "writer_lock_wait_ns_total",
        SQLITE_WRITER_LOCK_WAIT_NS_TOTAL
    );
    counter!("writer_lock_wait_ns_max", SQLITE_WRITER_LOCK_WAIT_NS_MAX);
    counter!("writer_lock_wait_ge_1ms", SQLITE_WRITER_LOCK_WAIT_GE_1MS);
    counter!("writer_lock_wait_ge_10ms", SQLITE_WRITER_LOCK_WAIT_GE_10MS);
    counter!(
        "writer_lock_wait_ge_100ms",
        SQLITE_WRITER_LOCK_WAIT_GE_100MS
    );
    counter!(
        "writer_lock_wait_ge_1000ms",
        SQLITE_WRITER_LOCK_WAIT_GE_1000MS
    );
    counter!(
        "writer_lock_held_ns_total",
        SQLITE_WRITER_LOCK_HELD_NS_TOTAL
    );
    counter!("writer_lock_held_ns_max", SQLITE_WRITER_LOCK_HELD_NS_MAX);
    counter!("writer_lock_held_ge_1ms", SQLITE_WRITER_LOCK_HELD_GE_1MS);
    counter!("writer_lock_held_ge_10ms", SQLITE_WRITER_LOCK_HELD_GE_10MS);
    counter!(
        "writer_lock_held_ge_100ms",
        SQLITE_WRITER_LOCK_HELD_GE_100MS
    );
    counter!(
        "writer_lock_held_ge_1000ms",
        SQLITE_WRITER_LOCK_HELD_GE_1000MS
    );
    counter!(
        "external_poll_data_version_reads",
        SQLITE_EXTERNAL_POLL_DATA_VERSION_READS
    );
    counter!(
        "external_poll_changed_table_reads",
        SQLITE_EXTERNAL_POLL_CHANGED_TABLE_READS
    );
    counter!(
        "external_poll_connection_opens",
        SQLITE_EXTERNAL_POLL_CONNECTION_OPENS
    );
    counter!(
        "external_poll_connection_open_failures",
        SQLITE_EXTERNAL_POLL_CONNECTION_OPEN_FAILURES
    );
    counter!("external_poll_wakeups", SQLITE_EXTERNAL_POLL_WAKEUPS);
    counter!(
        "external_poll_active_wakeups",
        SQLITE_EXTERNAL_POLL_ACTIVE_WAKEUPS
    );
    counter!(
        "external_poll_standby_wakeups",
        SQLITE_EXTERNAL_POLL_STANDBY_WAKEUPS
    );
    counter!(
        "external_poll_standby_entries",
        SQLITE_EXTERNAL_POLL_STANDBY_ENTRIES
    );
    counter!(
        "external_poll_active_resets",
        SQLITE_EXTERNAL_POLL_ACTIVE_RESETS
    );
    counter!(
        "external_poll_data_version_changes",
        SQLITE_EXTERNAL_POLL_DATA_VERSION_CHANGES
    );
    counter!(
        "external_poll_data_version_read_failures",
        SQLITE_EXTERNAL_POLL_DATA_VERSION_READ_FAILURES
    );
    counter!(
        "external_poll_changed_table_read_failures",
        SQLITE_EXTERNAL_POLL_CHANGED_TABLE_READ_FAILURES
    );
    counter!(
        "external_poll_changed_table_rows",
        SQLITE_EXTERNAL_POLL_CHANGED_TABLE_ROWS
    );
    counter!(
        "external_poll_changed_table_notifications",
        SQLITE_EXTERNAL_POLL_CHANGED_TABLE_NOTIFICATIONS
    );
    counter!(
        "external_poll_local_hook_suppressed_notifications",
        SQLITE_EXTERNAL_POLL_LOCAL_HOOK_SUPPRESSED_NOTIFICATIONS
    );
    counter!(
        "external_poll_drain_calls",
        SQLITE_EXTERNAL_POLL_DRAIN_CALLS
    );
    counter!(
        "external_poll_drain_batches",
        SQLITE_EXTERNAL_POLL_DRAIN_BATCHES
    );
    counter!(
        "external_poll_drain_empty_batches",
        SQLITE_EXTERNAL_POLL_DRAIN_EMPTY_BATCHES
    );
    counter!(
        "external_poll_drain_rows_visited",
        SQLITE_EXTERNAL_POLL_DRAIN_ROWS_VISITED
    );
    // Poll rows are decoded as they are visited; expose the compatibility key
    // from the single visited-row counter.
    counter!(
        "external_poll_drain_rows_decoded",
        SQLITE_EXTERNAL_POLL_DRAIN_ROWS_VISITED
    );
    counter!(
        "external_poll_drain_rows_max",
        SQLITE_EXTERNAL_POLL_DRAIN_ROWS_MAX
    );
    counter!(
        "external_poll_drain_batches_max",
        SQLITE_EXTERNAL_POLL_DRAIN_BATCHES_MAX
    );
    counter!(
        "external_poll_drain_budget_exhaustions",
        SQLITE_EXTERNAL_POLL_DRAIN_BUDGET_EXHAUSTIONS
    );
    counter!(
        "external_poll_drain_failures",
        SQLITE_EXTERNAL_POLL_DRAIN_FAILURES
    );
    out.insert(
        "external_poll_wakeups_by_database".to_string(),
        snapshot_counter_map(&SQLITE_EXTERNAL_POLL_WAKEUPS_BY_DATABASE),
    );
    out.insert(
        "external_poll_notifications_by_table".to_string(),
        snapshot_counter_map(&SQLITE_EXTERNAL_POLL_NOTIFICATIONS_BY_TABLE),
    );
    out.insert(
        "external_poll_local_hook_suppressions_by_table".to_string(),
        snapshot_counter_map(&SQLITE_EXTERNAL_POLL_LOCAL_HOOK_SUPPRESSIONS_BY_TABLE),
    );
    out.insert(
        "external_poll_drain_rows_by_table".to_string(),
        snapshot_counter_map(&SQLITE_EXTERNAL_POLL_DRAIN_ROWS_BY_TABLE),
    );
    out.insert(
        "external_poll_drain_batches_by_table".to_string(),
        snapshot_counter_map(&SQLITE_EXTERNAL_POLL_DRAIN_BATCHES_BY_TABLE),
    );
    out.insert(
        "external_poll_drain_budget_exhaustions_by_table".to_string(),
        snapshot_counter_map(&SQLITE_EXTERNAL_POLL_DRAIN_BUDGET_EXHAUSTIONS_BY_TABLE),
    );
    out.insert(
        "external_poll_drain_failures_by_table".to_string(),
        snapshot_counter_map(&SQLITE_EXTERNAL_POLL_DRAIN_FAILURES_BY_TABLE),
    );
    Value::Object(out)
}

fn snapshot_counter_map(map: &OnceLock<StdMutex<HashMap<String, u64>>>) -> Value {
    let Some(map) = map.get() else {
        return Value::Object(Default::default());
    };
    let counters = map.lock().unwrap();
    let sorted = counters
        .iter()
        .map(|(key, value)| (key.clone(), *value))
        .collect::<BTreeMap<_, _>>();
    serde_json::to_value(sorted).unwrap_or_else(|_| Value::Object(Default::default()))
}

fn snapshot_nested_counter_map(
    map: &OnceLock<StdMutex<HashMap<String, HashMap<String, u64>>>>,
) -> Value {
    let Some(map) = map.get() else {
        return Value::Object(Default::default());
    };
    let counters = map.lock().unwrap();
    let sorted = counters
        .iter()
        .map(|(outer, inner)| {
            let inner_sorted = inner
                .iter()
                .map(|(key, value)| (key.clone(), *value))
                .collect::<BTreeMap<_, _>>();
            (outer.clone(), inner_sorted)
        })
        .collect::<BTreeMap<_, _>>();
    serde_json::to_value(sorted).unwrap_or_else(|_| Value::Object(Default::default()))
}

fn increment_counter_map(map: &OnceLock<StdMutex<HashMap<String, u64>>>, key: &str) {
    increment_counter_map_by(map, key, 1);
}

fn increment_counter_map_by(
    map: &OnceLock<StdMutex<HashMap<String, u64>>>,
    key: &str,
    amount: u64,
) {
    if amount == 0 {
        return;
    }
    let mut counters = map
        .get_or_init(|| StdMutex::new(HashMap::new()))
        .lock()
        .unwrap();
    let counter = counters.entry(key.to_string()).or_insert(0);
    *counter = counter.saturating_add(amount);
}

fn increment_nested_counter_map(
    map: &OnceLock<StdMutex<HashMap<String, HashMap<String, u64>>>>,
    outer: &str,
    inner: &str,
) {
    increment_nested_counter_map_by(map, outer, inner, 1);
}

fn increment_nested_counter_map_by(
    map: &OnceLock<StdMutex<HashMap<String, HashMap<String, u64>>>>,
    outer: &str,
    inner: &str,
    amount: u64,
) {
    if amount == 0 {
        return;
    }
    let mut counters = map
        .get_or_init(|| StdMutex::new(HashMap::new()))
        .lock()
        .unwrap();
    let inner_counters = counters.entry(outer.to_string()).or_default();
    let counter = inner_counters.entry(inner.to_string()).or_insert(0);
    *counter = counter.saturating_add(amount);
}

fn normalized_query_fallback_operators(operator_families: &[String]) -> Vec<String> {
    if operator_families.is_empty() {
        vec!["$none".to_string()]
    } else {
        operator_families.to_vec()
    }
}

pub(crate) fn record_query_fallback_attribution(
    collection_name: &str,
    operator_families: &[String],
) {
    increment_counter_map(&SQLITE_QUERY_FALLBACK_BY_COLLECTION, collection_name);
    for operator in normalized_query_fallback_operators(operator_families) {
        increment_counter_map(&SQLITE_QUERY_FALLBACK_BY_OPERATOR, &operator);
        increment_nested_counter_map(
            &SQLITE_QUERY_FALLBACK_BY_COLLECTION_OPERATOR,
            collection_name,
            &operator,
        );
    }
}

pub(crate) fn record_query_fallback_rows(
    collection_name: &str,
    operator_families: &[String],
    rows_visited: u64,
) {
    SQLITE_QUERY_FALLBACK_ROWS_VISITED.fetch_add(rows_visited, Ordering::Relaxed);
    increment_counter_map_by(
        &SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_COLLECTION,
        collection_name,
        rows_visited,
    );
    for operator in normalized_query_fallback_operators(operator_families) {
        increment_counter_map_by(
            &SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_OPERATOR,
            &operator,
            rows_visited,
        );
        increment_nested_counter_map_by(
            &SQLITE_QUERY_FALLBACK_ROWS_VISITED_BY_COLLECTION_OPERATOR,
            collection_name,
            &operator,
            rows_visited,
        );
    }
}

pub(crate) struct TimedSqliteStatement {
    started: Instant,
}

impl Drop for TimedSqliteStatement {
    fn drop(&mut self) {
        record_sqlite_statement_elapsed(self.started.elapsed());
    }
}

pub(crate) fn timed_sqlite_statement() -> TimedSqliteStatement {
    SQLITE_STATEMENTS_EXECUTED.fetch_add(1, Ordering::Relaxed);
    TimedSqliteStatement {
        started: Instant::now(),
    }
}

pub(crate) fn record_sqlite_external_poll_data_version_read() {
    SQLITE_EXTERNAL_POLL_DATA_VERSION_READS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_changed_table_read() {
    SQLITE_EXTERNAL_POLL_CHANGED_TABLE_READS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_connection_open() {
    SQLITE_EXTERNAL_POLL_CONNECTION_OPENS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_connection_open_failure() {
    SQLITE_EXTERNAL_POLL_CONNECTION_OPEN_FAILURES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_wakeup(standby: bool, database_key: &str) {
    SQLITE_EXTERNAL_POLL_WAKEUPS.fetch_add(1, Ordering::Relaxed);
    increment_counter_map(&SQLITE_EXTERNAL_POLL_WAKEUPS_BY_DATABASE, database_key);
    if standby {
        SQLITE_EXTERNAL_POLL_STANDBY_WAKEUPS.fetch_add(1, Ordering::Relaxed);
    } else {
        SQLITE_EXTERNAL_POLL_ACTIVE_WAKEUPS.fetch_add(1, Ordering::Relaxed);
    }
}

pub(crate) fn record_sqlite_external_poll_standby_entry() {
    SQLITE_EXTERNAL_POLL_STANDBY_ENTRIES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_active_reset() {
    SQLITE_EXTERNAL_POLL_ACTIVE_RESETS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_data_version_change() {
    SQLITE_EXTERNAL_POLL_DATA_VERSION_CHANGES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_data_version_read_failure() {
    SQLITE_EXTERNAL_POLL_DATA_VERSION_READ_FAILURES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_changed_table_read_failure() {
    SQLITE_EXTERNAL_POLL_CHANGED_TABLE_READ_FAILURES.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_changed_table_rows(rows: usize) {
    SQLITE_EXTERNAL_POLL_CHANGED_TABLE_ROWS.fetch_add(rows as u64, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_external_poll_changed_table_notification(table_name: &str) {
    SQLITE_EXTERNAL_POLL_CHANGED_TABLE_NOTIFICATIONS.fetch_add(1, Ordering::Relaxed);
    increment_counter_map(&SQLITE_EXTERNAL_POLL_NOTIFICATIONS_BY_TABLE, table_name);
}

pub(crate) fn record_sqlite_external_poll_local_hook_suppression(table_name: &str) {
    SQLITE_EXTERNAL_POLL_LOCAL_HOOK_SUPPRESSED_NOTIFICATIONS.fetch_add(1, Ordering::Relaxed);
    increment_counter_map(
        &SQLITE_EXTERNAL_POLL_LOCAL_HOOK_SUPPRESSIONS_BY_TABLE,
        table_name,
    );
}

pub(crate) fn record_sqlite_external_poll_drain(
    table_name: &str,
    batches: usize,
    empty_batches: usize,
    rows: usize,
    drained_to_empty: bool,
) {
    SQLITE_EXTERNAL_POLL_DRAIN_CALLS.fetch_add(1, Ordering::Relaxed);
    SQLITE_EXTERNAL_POLL_DRAIN_BATCHES.fetch_add(batches as u64, Ordering::Relaxed);
    SQLITE_EXTERNAL_POLL_DRAIN_EMPTY_BATCHES.fetch_add(empty_batches as u64, Ordering::Relaxed);
    SQLITE_EXTERNAL_POLL_DRAIN_ROWS_VISITED.fetch_add(rows as u64, Ordering::Relaxed);
    update_atomic_max(&SQLITE_EXTERNAL_POLL_DRAIN_ROWS_MAX, rows as u64);
    update_atomic_max(&SQLITE_EXTERNAL_POLL_DRAIN_BATCHES_MAX, batches as u64);
    increment_counter_map_by(
        &SQLITE_EXTERNAL_POLL_DRAIN_ROWS_BY_TABLE,
        table_name,
        rows as u64,
    );
    increment_counter_map_by(
        &SQLITE_EXTERNAL_POLL_DRAIN_BATCHES_BY_TABLE,
        table_name,
        batches as u64,
    );
    if !drained_to_empty {
        SQLITE_EXTERNAL_POLL_DRAIN_BUDGET_EXHAUSTIONS.fetch_add(1, Ordering::Relaxed);
        increment_counter_map(
            &SQLITE_EXTERNAL_POLL_DRAIN_BUDGET_EXHAUSTIONS_BY_TABLE,
            table_name,
        );
    }
}

pub(crate) fn record_sqlite_external_poll_drain_failure(table_name: &str) {
    SQLITE_EXTERNAL_POLL_DRAIN_FAILURES.fetch_add(1, Ordering::Relaxed);
    increment_counter_map(&SQLITE_EXTERNAL_POLL_DRAIN_FAILURES_BY_TABLE, table_name);
}

pub(crate) fn record_sqlite_writer_lock_acquire() {
    SQLITE_WRITER_LOCK_ACQUIRE_CALLS.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn record_sqlite_writer_lock_wait(elapsed: Duration) {
    let elapsed_ns = duration_ns(elapsed);
    SQLITE_WRITER_LOCK_WAIT_NS_TOTAL.fetch_add(elapsed_ns, Ordering::Relaxed);
    update_atomic_max(&SQLITE_WRITER_LOCK_WAIT_NS_MAX, elapsed_ns);
    record_duration_buckets(
        elapsed_ns,
        &SQLITE_WRITER_LOCK_WAIT_GE_1MS,
        &SQLITE_WRITER_LOCK_WAIT_GE_10MS,
        &SQLITE_WRITER_LOCK_WAIT_GE_100MS,
        &SQLITE_WRITER_LOCK_WAIT_GE_1000MS,
    );
}

pub(crate) fn record_sqlite_writer_lock_held(elapsed: Duration) {
    let elapsed_ns = duration_ns(elapsed);
    SQLITE_WRITER_LOCK_HELD_NS_TOTAL.fetch_add(elapsed_ns, Ordering::Relaxed);
    update_atomic_max(&SQLITE_WRITER_LOCK_HELD_NS_MAX, elapsed_ns);
    record_duration_buckets(
        elapsed_ns,
        &SQLITE_WRITER_LOCK_HELD_GE_1MS,
        &SQLITE_WRITER_LOCK_HELD_GE_10MS,
        &SQLITE_WRITER_LOCK_HELD_GE_100MS,
        &SQLITE_WRITER_LOCK_HELD_GE_1000MS,
    );
}

fn record_sqlite_statement_elapsed(elapsed: Duration) {
    let elapsed_ns = duration_ns(elapsed);
    SQLITE_STATEMENT_ELAPSED_NS_TOTAL.fetch_add(elapsed_ns, Ordering::Relaxed);
    update_atomic_max(&SQLITE_STATEMENT_ELAPSED_NS_MAX, elapsed_ns);
    record_duration_buckets(
        elapsed_ns,
        &SQLITE_STATEMENT_ELAPSED_GE_1MS,
        &SQLITE_STATEMENT_ELAPSED_GE_10MS,
        &SQLITE_STATEMENT_ELAPSED_GE_100MS,
        &SQLITE_STATEMENT_ELAPSED_GE_1000MS,
    );
}

fn duration_ns(duration: Duration) -> u64 {
    (duration.as_nanos().min(u128::from(u64::MAX)) as u64).max(1)
}

fn record_duration_buckets(
    elapsed_ns: u64,
    ge_1ms: &AtomicU64,
    ge_10ms: &AtomicU64,
    ge_100ms: &AtomicU64,
    ge_1000ms: &AtomicU64,
) {
    if elapsed_ns >= 1_000_000 {
        ge_1ms.fetch_add(1, Ordering::Relaxed);
    }
    if elapsed_ns >= 10_000_000 {
        ge_10ms.fetch_add(1, Ordering::Relaxed);
    }
    if elapsed_ns >= 100_000_000 {
        ge_100ms.fetch_add(1, Ordering::Relaxed);
    }
    if elapsed_ns >= 1_000_000_000 {
        ge_1000ms.fetch_add(1, Ordering::Relaxed);
    }
}

fn update_atomic_max(value: &AtomicU64, candidate: u64) {
    let mut current = value.load(Ordering::Relaxed);
    while candidate > current {
        match value.compare_exchange_weak(current, candidate, Ordering::Relaxed, Ordering::Relaxed)
        {
            Ok(_) => break,
            Err(next_current) => current = next_current,
        }
    }
}

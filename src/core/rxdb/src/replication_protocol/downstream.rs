//! Port of `src/replication-protocol/downstream.ts`.
//!
//! Functional downstream replication protocol port for CTOX core.
//! - Conflict-aware initial sync (full 4-case decision in `persist_from_master`).
//! - Ongoing subscription to `replication_handler.master_change_stream()`:
//!   each `DocumentsWithCheckpoint` batch from the master is run through
//!   `persist_from_master`, while `"RESYNC"` sentinels trigger a paginated
//!   downstream resync.
//! - Per-batch `stream_queue.down` locking plus resync time cutoffs so
//!   already-covered master stream events cannot roll checkpoints back.
//! - Immediately buffered master stream document batches are coalesced into
//!   one `persist_from_master` call with stacked checkpoints. This is the Rust
//!   equivalent of RxDB's `addNewTask`/`openTasks` Promise-chain batching.
//! - Cancel handling: ongoing task aborts when `state.events.canceled` flips.
//! - `persist_from_master` deduplicates incoming master documents by primary key
//!   before writing, replacing RxDB's `nonPersistedFromMaster` Promise queue with
//!   a serialized, ownership-safe Rust path.

use std::collections::HashMap;
use std::sync::Arc;

use futures::{FutureExt, StreamExt};
use serde_json::{json, Value};

use crate::replication_protocol::checkpoint::{
    get_last_checkpoint_doc, set_checkpoint, set_initial_checkpoint,
};
use crate::replication_protocol::helper::{
    doc_state_to_write_doc, remote_revision_height_marker_matches,
    strip_attachments_data_from_meta_write_rows, wait_for_cancel, write_doc_to_doc_state,
    ReplicationClock,
};
use crate::replication_protocol::meta_instance::{get_assumed_master_state, get_meta_write_row};
use crate::rx_storage_helper::stack_checkpoints;
use crate::types::{
    BulkWriteRow, RxReplicationMasterChange, RxStorageInstanceReplicationState,
    RxStorageReplicationDirection,
};

// ref: rxdb/src/replication-protocol/downstream.ts:51-167
pub async fn start_replication_downstream(state: Arc<RxStorageInstanceReplicationState>) {
    // 1. Initial checkpoint write.
    let initial_checkpoint = state
        .input
        .initial_checkpoint
        .as_ref()
        .and_then(|initial| initial.downstream.as_ref());
    if let Err(e) = set_initial_checkpoint(
        &state,
        RxStorageReplicationDirection::Down,
        initial_checkpoint,
    )
    .await
    {
        tracing::error!(
            target: "ctox_rxdb::replication_protocol::downstream",
            "initial checkpoint write failed: {e}",
        );
        return;
    }

    // 2. Spawn ongoing master.change_stream subscription early.
    // Downstream uses `phase_start` as `last_time_master_changes_requested`:
    // master-stream events sequenced before the latest resync request are stale.
    let clock = Arc::new(ReplicationClock::default());
    let ongoing = spawn_ongoing_downstream(Arc::clone(&state), Arc::clone(&clock));

    // 3. Initial resync. A failed first pass must stay pending: consumers use
    // first_sync_done as the readiness invariant, so only a successful retry may
    // release waiters.
    let mut initial_sync_retry_delay = std::time::Duration::from_millis(50);
    while !state.events.canceled.get_value() {
        match downstream_resync_once(&state, &clock).await {
            Ok(()) => {
                if !state.first_sync_done.down.get_value() && !state.events.canceled.get_value() {
                    state.first_sync_done.down.next(true);
                }
                state
                    .events
                    .active
                    .finish_initial(RxStorageReplicationDirection::Down);
                break;
            }
            Err(e) => {
                state
                    .events
                    .error
                    .next(crate::plugins::utils::utils_error::error_to_plain_json(&e));
                tracing::error!(
                    target: "ctox_rxdb::replication_protocol::downstream",
                    "downstreamResyncOnce failed: {e}",
                );
                // Remote handler failures already apply the configured retry
                // delay, but local storage failures would otherwise spin hot.
                // Back off bounded and stay responsive to cancel.
                tokio::select! {
                    _ = tokio::time::sleep(initial_sync_retry_delay) => {}
                    _ = wait_for_cancel(&state) => break,
                }
                initial_sync_retry_delay =
                    (initial_sync_retry_delay * 2).min(std::time::Duration::from_secs(5));
            }
        }
    }

    // 4. Stay alive until canceled, then drop the subscription.
    wait_for_cancel(&state).await;
    ongoing.abort();
}

/// Spawn the long-lived task that consumes `replication_handler.master_change_stream()`
/// and writes each batch to the fork via `persist_from_master`.
fn spawn_ongoing_downstream(
    state: Arc<RxStorageInstanceReplicationState>,
    clock: Arc<ReplicationClock>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut stream = state.input.replication_handler.master_change_stream();
        while let Some(master_change) = stream.next().await {
            if state.events.canceled.get_value() {
                break;
            }
            // Mark the event pending before it waits on upstream. Otherwise a
            // whole-replication idle waiter could miss this hand-off window.
            let _activity = state
                .events
                .active
                .track(RxStorageReplicationDirection::Down);
            let task_time = clock.next_time();
            wait_until_upstream_inactive(&state).await;
            if task_time < clock.phase_start() {
                continue;
            }
            match master_change {
                RxReplicationMasterChange::Resync => {
                    run_downstream_resync_from_stream(&state, &clock).await;
                }
                RxReplicationMasterChange::Documents(docs_with_cp) => {
                    let mut tasks = vec![(task_time, docs_with_cp)];
                    let mut run_resync_after_batch = false;
                    while let Some(Some(next_master_change)) = stream.next().now_or_never() {
                        if state.events.canceled.get_value() {
                            break;
                        }
                        let next_task_time = clock.next_time();
                        match next_master_change {
                            RxReplicationMasterChange::Documents(next_docs_with_cp) => {
                                tasks.push((next_task_time, next_docs_with_cp));
                            }
                            RxReplicationMasterChange::Resync => {
                                run_resync_after_batch = true;
                                break;
                            }
                        }
                    }
                    if let Err(e) =
                        process_downstream_master_change_tasks(&state, &clock, tasks).await
                    {
                        tracing::error!(
                            target: "ctox_rxdb::replication_protocol::downstream",
                            "ongoing persist_from_master failed: {e}",
                        );
                    }
                    if run_resync_after_batch {
                        run_downstream_resync_from_stream(&state, &clock).await;
                    }
                }
            }
        }
    })
}

async fn process_downstream_master_change_tasks(
    state: &Arc<RxStorageInstanceReplicationState>,
    clock: &ReplicationClock,
    tasks: Vec<(i64, crate::types::DocumentsWithCheckpoint)>,
) -> Result<(), crate::rx_error::RxError> {
    let cutoff = clock.phase_start();
    let mut docs = Vec::new();
    let mut checkpoints = Vec::new();
    let mut emit_count = 0;
    for (task_time, docs_with_cp) in tasks {
        if task_time < cutoff || docs_with_cp.documents.is_empty() {
            continue;
        }
        emit_count += 1;
        docs.extend(docs_with_cp.documents);
        checkpoints.push(Some(docs_with_cp.checkpoint));
    }
    if docs.is_empty() && checkpoints.is_empty() {
        return Ok(());
    }
    // Both directions collect accepted task checkpoints and stack exactly once.
    // The merge is order-preserving, so the newest value for each shard wins
    // without direction-specific intermediate checkpoint shapes.
    let checkpoint = stack_checkpoints(&checkpoints);
    // Count source stream emissions, not coalesced drains. A buffered drain can
    // contain multiple accepted tasks and uses the same per-task semantics in
    // both replication directions.
    {
        let mut stats = state.stats.down.lock();
        stats.master_change_stream_emit += emit_count;
    }
    let _g = state.stream_queue.down.lock().await;
    persist_from_master(state, docs, checkpoint).await
}

async fn run_downstream_resync_from_stream(
    state: &Arc<RxStorageInstanceReplicationState>,
    clock: &ReplicationClock,
) {
    if let Err(e) = downstream_resync_once(state, clock).await {
        tracing::error!(
            target: "ctox_rxdb::replication_protocol::downstream",
            "RESYNC downstreamResyncOnce failed: {e}",
        );
    }
}

async fn wait_until_upstream_inactive(state: &RxStorageInstanceReplicationState) {
    crate::replication_protocol::index_mod::await_rx_storage_replication_direction_idle(
        state,
        RxStorageReplicationDirection::Up,
        crate::replication_protocol::index_mod::ReplicationIdleRequirement::ActivityAndQueue,
    )
    .await;
}

// ref: rxdb/src/replication-protocol/downstream.ts:175-217
async fn downstream_resync_once(
    state: &Arc<RxStorageInstanceReplicationState>,
    clock: &ReplicationClock,
) -> Result<(), crate::rx_error::RxError> {
    {
        let mut stats = state.stats.down.lock();
        stats.downstream_resync_once += 1;
    }
    if state.events.canceled.get_value() {
        return Ok(());
    }
    let last_checkpoint_doc =
        get_last_checkpoint_doc(state, RxStorageReplicationDirection::Down).await?;
    let mut last_checkpoint: Value = last_checkpoint_doc.unwrap_or(Value::Null);
    let pull_batch_size = state.input.pull_batch_size;

    while !state.events.canceled.get_value() {
        // Acquire stream_queue.down per batch so ongoing events can interleave.
        let _g = state.stream_queue.down.lock().await;
        let request_time = clock.next_time();
        clock.set_phase_start(request_time);
        let down_result = state
            .input
            .replication_handler
            .master_changes_since(
                if last_checkpoint.is_null() {
                    None
                } else {
                    Some(last_checkpoint.clone())
                },
                pull_batch_size,
            )
            .await?;
        if down_result.documents.is_empty() {
            break;
        }
        last_checkpoint = stack_checkpoints(&[
            if last_checkpoint.is_null() {
                None
            } else {
                Some(last_checkpoint.clone())
            },
            Some(down_result.checkpoint.clone()),
        ]);
        persist_from_master(
            state,
            down_result.documents.clone(),
            last_checkpoint.clone(),
        )
        .await?;
        let small = (down_result.documents.len() as u64) < pull_batch_size;
        drop(_g);
        if small {
            break;
        }
    }
    Ok(())
}

// ref: rxdb/src/replication-protocol/downstream.ts:255-end (persistFromMaster, FULL conflict-aware version)
async fn persist_from_master(
    state: &Arc<RxStorageInstanceReplicationState>,
    docs: Vec<Value>,
    new_down_checkpoint: Value,
) -> Result<(), crate::rx_error::RxError> {
    {
        let mut stats = state.stats.down.lock();
        stats.persist_from_master += 1;
    }
    if docs.is_empty() {
        return set_checkpoint(
            state,
            RxStorageReplicationDirection::Down,
            new_down_checkpoint,
        )
        .await;
    }

    let primary_path = state.primary_path.clone();
    let has_attachments = state.has_attachments;
    let keep_meta = state.input.keep_meta;

    // Build downDocsById for quick lookup.
    let mut down_docs_by_id: HashMap<String, Value> = HashMap::new();
    for d in docs.into_iter() {
        if let Some(id) = d.get(&primary_path).and_then(|v| v.as_str()) {
            down_docs_by_id.insert(id.to_string(), d);
        }
    }
    let doc_ids: Vec<String> = down_docs_by_id.keys().cloned().collect();

    // Read fork state + assumed-master state in parallel.
    let (fork_state_list, assumed_master_state) = tokio::join!(
        state
            .input
            .fork_instance
            .find_documents_by_id(&doc_ids, true),
        get_assumed_master_state(state, &doc_ids),
    );
    let fork_state_list = fork_state_list?;
    let assumed_master_state = assumed_master_state?;

    let mut fork_state_by_id: HashMap<String, Value> = HashMap::new();
    for ex in fork_state_list.into_iter() {
        if let Some(id) = ex.get(&primary_path).and_then(|v| v.as_str()) {
            fork_state_by_id.insert(id.to_string(), ex);
        }
    }

    let mut write_rows_to_fork: Vec<BulkWriteRow> = Vec::new();
    let mut write_rows_to_meta: Vec<BulkWriteRow> = Vec::new();

    for (doc_id, master_doc_state) in down_docs_by_id.iter() {
        let fork_state = fork_state_by_id.get(doc_id).cloned();
        let assumed_master = assumed_master_state.get(doc_id).cloned();
        let prepared_master = doc_state_to_write_doc(
            &state.checkpoint_key,
            has_attachments,
            keep_meta,
            master_doc_state,
            None,
        );

        match (fork_state, assumed_master) {
            (None, _) => {
                // No fork doc — straight insert.
                write_rows_to_fork.push(BulkWriteRow {
                    previous: None,
                    document: prepared_master.clone(),
                });
                let meta_row = get_meta_write_row(
                    state,
                    &write_doc_to_doc_state(&prepared_master, has_attachments, keep_meta),
                    None,
                    None,
                )
                .await?;
                write_rows_to_meta.push(meta_row);
            }
            (Some(fork), Some(asm)) => {
                if asm
                    .meta_document
                    .get("isResolvedConflict")
                    .and_then(Value::as_str)
                    == fork.get("_rev").and_then(Value::as_str)
                {
                    // This is deliberately queue-head semantics: upstream RxDB
                    // awaits `streamQueue.up` here so the resolved write is sent,
                    // without waiting for unrelated future upstream activity.
                    crate::replication_protocol::index_mod::await_rx_storage_replication_direction_idle(
                        state,
                        RxStorageReplicationDirection::Up,
                        crate::replication_protocol::index_mod::ReplicationIdleRequirement::QueueOnly,
                    )
                    .await;
                }
                let fork_clean = write_doc_to_doc_state(&fork, has_attachments, keep_meta);
                let master_clean =
                    write_doc_to_doc_state(master_doc_state, has_attachments, keep_meta);
                let already_equal = state
                    .input
                    .conflict_handler
                    .is_equal(&fork_clean, &master_clean, "downstream-already-equal")
                    .await;
                if already_equal {
                    // Skip — fork already mirrors master. Refresh meta-doc for
                    // bookkeeping (assumed_master_state.docData = master).
                    let meta_row =
                        get_meta_write_row(state, &master_clean, Some(&asm.meta_document), None)
                            .await?;
                    write_rows_to_meta.push(meta_row);
                    continue;
                }
                // Compare against assumed master: did the fork diverge?
                let mut fork_matches_assumed = state
                    .input
                    .conflict_handler
                    .is_equal(&fork_clean, &asm.doc_data, "downstream-vs-assumed")
                    .await;
                if !fork_matches_assumed
                    && asm.doc_data.get("_rev").is_some()
                    && remote_revision_height_marker_matches(&fork, &state.input.identifier)
                {
                    fork_matches_assumed = true;
                }
                if fork_matches_assumed {
                    // No conflict — fast-forward fork to master.
                    write_rows_to_fork.push(BulkWriteRow {
                        previous: Some(fork.clone()),
                        document: doc_state_to_write_doc(
                            &state.checkpoint_key,
                            has_attachments,
                            keep_meta,
                            master_doc_state,
                            Some(&fork),
                        ),
                    });
                    let meta_row =
                        get_meta_write_row(state, &master_clean, Some(&asm.meta_document), None)
                            .await?;
                    write_rows_to_meta.push(meta_row);
                } else {
                    // Local fork has unpushed changes. Downstream must not
                    // resolve or overwrite them; upstream will resolve later.
                    continue;
                }
            }
            (Some(_fork), None) => {
                // Existing local doc without assumed-master state is an
                // unreplicated local write. Skip downstream overwrite.
                continue;
            }
        }
    }

    if !write_rows_to_fork.is_empty() {
        let context = state.downstream_bulk_write_flag.clone();
        let write_rows_for_emit = write_rows_to_fork.clone();
        let result = state
            .input
            .fork_instance
            .bulk_write(write_rows_to_fork, &context)
            .await?;
        let failed_ids: std::collections::HashSet<String> = result
            .error
            .iter()
            .map(|error| error.document_id.clone())
            .collect();
        for row in write_rows_for_emit.iter() {
            let Some(id) = row
                .document
                .get(&primary_path)
                .and_then(|value| value.as_str())
            else {
                continue;
            };
            if failed_ids.contains(id) {
                continue;
            }
            state.events.processed.down.next(json!({
                "document": write_doc_to_doc_state(&row.document, has_attachments, keep_meta),
            }));
        }
    }
    if !write_rows_to_meta.is_empty() {
        let _ = state
            .input
            .meta_instance
            .bulk_write(
                strip_attachments_data_from_meta_write_rows(state, &write_rows_to_meta),
                "replication-meta-write",
            )
            .await?;
    }

    set_checkpoint(
        state,
        RxStorageReplicationDirection::Down,
        new_down_checkpoint,
    )
    .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use serde_json::json;

    use crate::plugins::storage_memory::get_rx_storage_memory;
    use crate::replication_protocol::index_mod::test_utils::{
        test_schema, ReplicationStateBuilder,
    };
    use crate::replication_protocol::meta_instance::get_rx_replication_meta_instance_schema;
    use crate::rxjs_compat::{RxStream, RxSubject};
    use crate::types::{
        DocumentsWithCheckpoint, RxReplicationHandler, RxReplicationMasterChange,
        RxStorageInstance, RxStorageInstanceCreationParams, RxStorageReplicationDirection,
    };

    struct NoopReplicationHandler;

    #[async_trait]
    impl RxReplicationHandler for NoopReplicationHandler {
        fn master_change_stream(&self) -> RxStream<RxReplicationMasterChange> {
            Box::pin(tokio_stream::empty())
        }

        async fn master_changes_since(
            &self,
            _checkpoint: Option<Value>,
            _batch_size: u64,
        ) -> Result<DocumentsWithCheckpoint, crate::rx_error::RxError> {
            Ok(DocumentsWithCheckpoint {
                documents: Vec::new(),
                checkpoint: Value::Null,
            })
        }

        async fn master_write(
            &self,
            _rows: Vec<crate::types::RxReplicationWriteToMasterRow>,
        ) -> Result<Vec<Value>, crate::rx_error::RxError> {
            Ok(Vec::new())
        }
    }

    struct InitialSyncRetryHandler {
        master_changes_since_calls: Arc<AtomicUsize>,
        allow_second_attempt: Arc<tokio::sync::Notify>,
    }

    #[async_trait]
    impl RxReplicationHandler for InitialSyncRetryHandler {
        fn master_change_stream(&self) -> RxStream<RxReplicationMasterChange> {
            Box::pin(tokio_stream::empty())
        }

        async fn master_changes_since(
            &self,
            _checkpoint: Option<Value>,
            _batch_size: u64,
        ) -> Result<DocumentsWithCheckpoint, crate::rx_error::RxError> {
            let attempt = self
                .master_changes_since_calls
                .fetch_add(1, Ordering::SeqCst);
            if attempt == 0 {
                return Err(crate::rx_error::new_rx_error(
                    "TEST_DOWNSTREAM_INITIAL_SYNC",
                    Some(json!({ "attempt": 1 })),
                ));
            }
            self.allow_second_attempt.notified().await;
            Ok(DocumentsWithCheckpoint {
                documents: Vec::new(),
                checkpoint: Value::Null,
            })
        }

        async fn master_write(
            &self,
            _rows: Vec<crate::types::RxReplicationWriteToMasterRow>,
        ) -> Result<Vec<Value>, crate::rx_error::RxError> {
            Ok(Vec::new())
        }
    }

    struct StreamReplicationHandler {
        stream: RxSubject<RxReplicationMasterChange>,
    }

    #[async_trait]
    impl RxReplicationHandler for StreamReplicationHandler {
        fn master_change_stream(&self) -> RxStream<RxReplicationMasterChange> {
            self.stream.subscribe()
        }

        async fn master_changes_since(
            &self,
            _checkpoint: Option<Value>,
            _batch_size: u64,
        ) -> Result<DocumentsWithCheckpoint, crate::rx_error::RxError> {
            Ok(DocumentsWithCheckpoint {
                documents: Vec::new(),
                checkpoint: Value::Null,
            })
        }

        async fn master_write(
            &self,
            _rows: Vec<crate::types::RxReplicationWriteToMasterRow>,
        ) -> Result<Vec<Value>, crate::rx_error::RxError> {
            Ok(Vec::new())
        }
    }

    struct ResyncPullReplicationHandler {
        stream: RxSubject<RxReplicationMasterChange>,
        master_changes_since_calls: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl RxReplicationHandler for ResyncPullReplicationHandler {
        fn master_change_stream(&self) -> RxStream<RxReplicationMasterChange> {
            self.stream.subscribe()
        }

        async fn master_changes_since(
            &self,
            _checkpoint: Option<Value>,
            _batch_size: u64,
        ) -> Result<DocumentsWithCheckpoint, crate::rx_error::RxError> {
            if self
                .master_changes_since_calls
                .fetch_add(1, Ordering::SeqCst)
                == 0
            {
                Ok(DocumentsWithCheckpoint {
                    documents: vec![json!({
                        "id": "a",
                        "age": 7,
                        "_deleted": false
                    })],
                    checkpoint: json!({ "sequence": 7 }),
                })
            } else {
                Ok(DocumentsWithCheckpoint {
                    documents: Vec::new(),
                    checkpoint: json!({ "sequence": 7 }),
                })
            }
        }

        async fn master_write(
            &self,
            _rows: Vec<crate::types::RxReplicationWriteToMasterRow>,
        ) -> Result<Vec<Value>, crate::rx_error::RxError> {
            Ok(Vec::new())
        }
    }

    fn fork_doc(age: i64, rev: &str) -> Value {
        json!({
            "id": "a",
            "age": age,
            "_deleted": false,
            "_attachments": {},
            "_rev": rev,
            "_meta": { "lwt": 1.0 },
        })
    }

    #[tokio::test]
    async fn initial_downstream_sync_error_stays_pending_relays_error_and_retries() {
        let master_changes_since_calls = Arc::new(AtomicUsize::new(0));
        let allow_second_attempt = Arc::new(tokio::sync::Notify::new());
        let state = ReplicationStateBuilder::new(
            "db-downstream-initial-error-retry",
            Arc::new(InitialSyncRetryHandler {
                master_changes_since_calls: Arc::clone(&master_changes_since_calls),
                allow_second_attempt: Arc::clone(&allow_second_attempt),
            }),
        )
        .build()
        .await
        .state;
        let mut errors = state.events.error.subscribe();
        let state_for_task = Arc::clone(&state);
        let replication_task =
            tokio::spawn(async move { start_replication_downstream(state_for_task).await });

        let error = tokio::time::timeout(std::time::Duration::from_secs(1), errors.next())
            .await
            .unwrap()
            .unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while master_changes_since_calls.load(Ordering::SeqCst) < 2 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();

        assert_eq!(error["code"], json!("TEST_DOWNSTREAM_INITIAL_SYNC"));
        assert!(!state.first_sync_done.down.get_value());
        assert_eq!(master_changes_since_calls.load(Ordering::SeqCst), 2);

        allow_second_attempt.notify_one();
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while !state.first_sync_done.down.get_value() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .unwrap();
        state.events.canceled.next(true);
        tokio::time::timeout(std::time::Duration::from_secs(1), replication_task)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn downstream_skips_unpushed_local_fork_conflicts() {
        let storage = get_rx_storage_memory(());
        let schema = test_schema();
        let fork_instance: Arc<dyn RxStorageInstance> = storage
            .create_storage_instance(
                RxStorageInstanceCreationParams {
                    database_instance_token: "db-token".to_string(),
                    database_name: "db-downstream-conflict".to_string(),
                    collection_name: "docs".to_string(),
                    schema: schema.clone(),
                    options: HashMap::new(),
                    multi_instance: false,
                    dev_mode: false,
                    password: None,
                },
                (),
            )
            .await
            .unwrap();
        fork_instance
            .bulk_write(
                vec![BulkWriteRow {
                    previous: None,
                    document: fork_doc(2, "1-local"),
                }],
                "seed-local",
            )
            .await
            .unwrap();
        let meta_schema = get_rx_replication_meta_instance_schema(&schema, false).unwrap();
        let meta_instance: Arc<dyn RxStorageInstance> = storage
            .create_storage_instance(
                RxStorageInstanceCreationParams {
                    database_instance_token: "db-token".to_string(),
                    database_name: "db-downstream-conflict".to_string(),
                    collection_name: "meta".to_string(),
                    schema: meta_schema,
                    options: HashMap::new(),
                    multi_instance: false,
                    dev_mode: false,
                    password: None,
                },
                (),
            )
            .await
            .unwrap();
        let state = ReplicationStateBuilder::from_instances(
            Arc::clone(&fork_instance),
            Arc::clone(&meta_instance),
            Arc::new(NoopReplicationHandler),
        )
        .build()
        .await
        .state;
        let assumed_master = json!({
            "id": "a",
            "age": 1,
            "_deleted": false
        });
        let meta_row = get_meta_write_row(&state, &assumed_master, None, None)
            .await
            .unwrap();
        meta_instance
            .bulk_write(vec![meta_row], "seed-assumed-master")
            .await
            .unwrap();

        persist_from_master(
            &state,
            vec![json!({
                "id": "a",
                "age": 3,
                "_deleted": false
            })],
            json!({ "sequence": 1 }),
        )
        .await
        .unwrap();

        let fork_docs = fork_instance
            .find_documents_by_id(&["a".to_string()], true)
            .await
            .unwrap();
        assert_eq!(fork_docs[0]["age"], json!(2));
        let assumed_after = get_assumed_master_state(&state, &["a".to_string()])
            .await
            .unwrap();
        assert_eq!(assumed_after["a"].doc_data["age"], json!(1));
        let checkpoint = get_last_checkpoint_doc(&state, RxStorageReplicationDirection::Down)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(checkpoint["sequence"], json!(1));
    }

    #[tokio::test]
    async fn downstream_fast_forwards_remote_revision_marked_fork_state() {
        let storage = get_rx_storage_memory(());
        let schema = test_schema();
        let fork_instance: Arc<dyn RxStorageInstance> = storage
            .create_storage_instance(
                RxStorageInstanceCreationParams {
                    database_instance_token: "db-token".to_string(),
                    database_name: "db-downstream-rev-height".to_string(),
                    collection_name: "docs".to_string(),
                    schema: schema.clone(),
                    options: HashMap::new(),
                    multi_instance: false,
                    dev_mode: false,
                    password: None,
                },
                (),
            )
            .await
            .unwrap();
        fork_instance
            .bulk_write(
                vec![BulkWriteRow {
                    previous: None,
                    document: json!({
                        "id": "a",
                        "age": 2,
                        "_deleted": false,
                        "_attachments": {},
                        "_rev": "2-local",
                        "_meta": {
                            "lwt": 1.0,
                            "replication-test": 2
                        },
                    }),
                }],
                "seed-local",
            )
            .await
            .unwrap();
        let meta_schema = get_rx_replication_meta_instance_schema(&schema, false).unwrap();
        let meta_instance: Arc<dyn RxStorageInstance> = storage
            .create_storage_instance(
                RxStorageInstanceCreationParams {
                    database_instance_token: "db-token".to_string(),
                    database_name: "db-downstream-rev-height".to_string(),
                    collection_name: "meta".to_string(),
                    schema: meta_schema,
                    options: HashMap::new(),
                    multi_instance: false,
                    dev_mode: false,
                    password: None,
                },
                (),
            )
            .await
            .unwrap();
        let state = ReplicationStateBuilder::from_instances(
            Arc::clone(&fork_instance),
            Arc::clone(&meta_instance),
            Arc::new(NoopReplicationHandler),
        )
        .build()
        .await
        .state;
        let assumed_master = json!({
            "id": "a",
            "age": 1,
            "_deleted": false,
            "_rev": "2-master"
        });
        let meta_row = get_meta_write_row(&state, &assumed_master, None, None)
            .await
            .unwrap();
        meta_instance
            .bulk_write(vec![meta_row], "seed-assumed-master")
            .await
            .unwrap();

        persist_from_master(
            &state,
            vec![json!({
                "id": "a",
                "age": 3,
                "_deleted": false,
                "_rev": "3-master"
            })],
            json!({ "sequence": 2 }),
        )
        .await
        .unwrap();

        let fork_docs = fork_instance
            .find_documents_by_id(&["a".to_string()], true)
            .await
            .unwrap();
        assert_eq!(fork_docs[0]["age"], json!(3));
        let assumed_after = get_assumed_master_state(&state, &["a".to_string()])
            .await
            .unwrap();
        assert_eq!(assumed_after["a"].doc_data["age"], json!(3));
    }

    #[tokio::test]
    async fn downstream_waits_for_upstream_queue_when_fork_is_resolved_conflict() {
        let storage = get_rx_storage_memory(());
        let schema = test_schema();
        let fork_instance: Arc<dyn RxStorageInstance> = storage
            .create_storage_instance(
                RxStorageInstanceCreationParams {
                    database_instance_token: "db-token".to_string(),
                    database_name: "db-downstream-resolved-conflict-wait".to_string(),
                    collection_name: "docs".to_string(),
                    schema: schema.clone(),
                    options: HashMap::new(),
                    multi_instance: false,
                    dev_mode: false,
                    password: None,
                },
                (),
            )
            .await
            .unwrap();
        fork_instance
            .bulk_write(
                vec![BulkWriteRow {
                    previous: None,
                    document: fork_doc(2, "2-resolved"),
                }],
                "seed-local",
            )
            .await
            .unwrap();
        let meta_schema = get_rx_replication_meta_instance_schema(&schema, false).unwrap();
        let meta_instance: Arc<dyn RxStorageInstance> = storage
            .create_storage_instance(
                RxStorageInstanceCreationParams {
                    database_instance_token: "db-token".to_string(),
                    database_name: "db-downstream-resolved-conflict-wait".to_string(),
                    collection_name: "meta".to_string(),
                    schema: meta_schema,
                    options: HashMap::new(),
                    multi_instance: false,
                    dev_mode: false,
                    password: None,
                },
                (),
            )
            .await
            .unwrap();
        let state = ReplicationStateBuilder::from_instances(
            Arc::clone(&fork_instance),
            Arc::clone(&meta_instance),
            Arc::new(NoopReplicationHandler),
        )
        .build()
        .await
        .state;
        let assumed_master = json!({
            "id": "a",
            "age": 1,
            "_deleted": false
        });
        let meta_row = get_meta_write_row(&state, &assumed_master, None, Some("2-resolved"))
            .await
            .unwrap();
        meta_instance
            .bulk_write(vec![meta_row], "seed-assumed-master")
            .await
            .unwrap();

        let up_queue_guard = state.stream_queue.up.lock().await;
        let state_for_task = Arc::clone(&state);
        let persist_task = tokio::spawn(async move {
            persist_from_master(
                &state_for_task,
                vec![json!({
                    "id": "a",
                    "age": 3,
                    "_deleted": false
                })],
                json!({ "sequence": 3 }),
            )
            .await
            .unwrap();
        });

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        let fork_docs = fork_instance
            .find_documents_by_id(&["a".to_string()], true)
            .await
            .unwrap();
        assert_eq!(fork_docs[0]["age"], json!(2));
        assert!(!persist_task.is_finished());

        drop(up_queue_guard);
        tokio::time::timeout(std::time::Duration::from_secs(1), persist_task)
            .await
            .unwrap()
            .unwrap();
        let fork_docs = fork_instance
            .find_documents_by_id(&["a".to_string()], true)
            .await
            .unwrap();
        assert_eq!(fork_docs[0]["age"], json!(2));
        let checkpoint = get_last_checkpoint_doc(&state, RxStorageReplicationDirection::Down)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(checkpoint["sequence"], json!(3));
    }

    #[tokio::test]
    async fn downstream_master_stream_waits_until_upstream_inactive() {
        let master_stream = RxSubject::new();
        let built = ReplicationStateBuilder::new(
            "db-downstream-active-up",
            Arc::new(StreamReplicationHandler {
                stream: master_stream.clone(),
            }),
        )
        .build()
        .await;
        let state = built.state;
        let fork_instance = built.fork_instance;
        state.events.active.up.next(true);
        let ongoing =
            spawn_ongoing_downstream(Arc::clone(&state), Arc::new(ReplicationClock::default()));
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        master_stream.next(RxReplicationMasterChange::Documents(
            DocumentsWithCheckpoint {
                documents: vec![json!({
                    "id": "a",
                    "age": 3,
                    "_deleted": false
                })],
                checkpoint: json!({ "sequence": 1 }),
            },
        ));

        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
        assert!(fork_instance
            .find_documents_by_id(&["a".to_string()], true)
            .await
            .unwrap()
            .is_empty());

        state.events.active.up.next(false);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let docs = fork_instance
                    .find_documents_by_id(&["a".to_string()], true)
                    .await
                    .unwrap();
                if !docs.is_empty() {
                    assert_eq!(docs[0]["age"], json!(3));
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        ongoing.abort();
    }

    #[tokio::test]
    async fn ongoing_downstream_batches_buffered_master_events() {
        let master_stream = RxSubject::new();
        let built = ReplicationStateBuilder::new(
            "db-downstream-buffered-batch",
            Arc::new(StreamReplicationHandler {
                stream: master_stream.clone(),
            }),
        )
        .build()
        .await;
        let state = built.state;
        let fork_instance = built.fork_instance;
        state.events.active.up.next(true);
        let ongoing =
            spawn_ongoing_downstream(Arc::clone(&state), Arc::new(ReplicationClock::default()));
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        master_stream.next(RxReplicationMasterChange::Documents(
            DocumentsWithCheckpoint {
                documents: vec![json!({
                    "id": "a",
                    "age": 3,
                    "_deleted": false
                })],
                checkpoint: json!({ "sequence": 1 }),
            },
        ));
        master_stream.next(RxReplicationMasterChange::Documents(
            DocumentsWithCheckpoint {
                documents: vec![json!({
                    "id": "b",
                    "age": 4,
                    "_deleted": false
                })],
                checkpoint: json!({ "sequence": 2 }),
            },
        ));

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert!(fork_instance
            .find_documents_by_id(&["a".to_string(), "b".to_string()], true)
            .await
            .unwrap()
            .is_empty());

        state.events.active.up.next(false);
        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let docs = fork_instance
                    .find_documents_by_id(&["a".to_string(), "b".to_string()], true)
                    .await
                    .unwrap();
                if docs.len() == 2 {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();

        {
            let stats = state.stats.down.lock();
            assert_eq!(stats.master_change_stream_emit, 2);
            assert_eq!(stats.persist_from_master, 1);
        }
        let checkpoint = get_last_checkpoint_doc(&state, RxStorageReplicationDirection::Down)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(checkpoint["sequence"], json!(2));
        ongoing.abort();
    }

    #[tokio::test]
    async fn downstream_master_stream_skips_events_covered_by_resync_cutoff() {
        let master_stream = RxSubject::new();
        let built = ReplicationStateBuilder::new(
            "db-downstream-cutoff",
            Arc::new(StreamReplicationHandler {
                stream: master_stream.clone(),
            }),
        )
        .build()
        .await;
        let state = built.state;
        let fork_instance = built.fork_instance;
        let clock = Arc::new(ReplicationClock::new(0, 1));
        let ongoing = spawn_ongoing_downstream(Arc::clone(&state), Arc::clone(&clock));
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        master_stream.next(RxReplicationMasterChange::Documents(
            DocumentsWithCheckpoint {
                documents: vec![json!({
                    "id": "a",
                    "age": 3,
                    "_deleted": false
                })],
                checkpoint: json!({ "sequence": 1 }),
            },
        ));

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            while clock.time() == 0 {
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        ongoing.abort();

        assert!(fork_instance
            .find_documents_by_id(&["a".to_string()], true)
            .await
            .unwrap()
            .is_empty());
        assert!(
            get_last_checkpoint_doc(&state, RxStorageReplicationDirection::Down)
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn master_change_stream_resync_triggers_downstream_resync() {
        let master_stream = RxSubject::new();
        let master_changes_since_calls = Arc::new(AtomicUsize::new(0));
        let built = ReplicationStateBuilder::new(
            "db-downstream-resync",
            Arc::new(ResyncPullReplicationHandler {
                stream: master_stream.clone(),
                master_changes_since_calls: Arc::clone(&master_changes_since_calls),
            }),
        )
        .build()
        .await;
        let state = built.state;
        let fork_instance = built.fork_instance;
        state.events.active.up.next(false);
        let ongoing =
            spawn_ongoing_downstream(Arc::clone(&state), Arc::new(ReplicationClock::default()));
        tokio::time::sleep(std::time::Duration::from_millis(10)).await;

        master_stream.next(RxReplicationMasterChange::Resync);

        tokio::time::timeout(std::time::Duration::from_secs(1), async {
            loop {
                let docs = fork_instance
                    .find_documents_by_id(&["a".to_string()], true)
                    .await
                    .unwrap();
                if docs.first().and_then(|doc| doc.get("age")) == Some(&json!(7)) {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            }
        })
        .await
        .unwrap();
        assert!(master_changes_since_calls.load(Ordering::SeqCst) > 0);
        ongoing.abort();
    }
}

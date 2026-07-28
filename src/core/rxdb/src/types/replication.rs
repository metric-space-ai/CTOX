//! Replication types — port of the relevant entries from
//! `rxdb/src/types/{replication-protocol,replication}.d.ts`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::Mutex;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use serde_json::Value;
use tokio::sync::{Mutex as TokioMutex, MutexGuard as TokioMutexGuard};

use crate::rx_error::RxError;
use crate::rxjs_compat::{RxBehaviorSubject, RxStream, RxSubject};
use crate::types::{RxStorageInstance, SharedHashFunction};

// ref: rxdb/src/types/replication-protocol.d.ts RxStorageReplicationDirection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize, Serialize)]
pub enum RxStorageReplicationDirection {
    Up,
    Down,
}

impl RxStorageReplicationDirection {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Up => "up",
            Self::Down => "down",
        }
    }
}

// ref: rxdb/src/types/replication-protocol.d.ts RxConflictHandlerInput<RxDocType>
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RxConflictHandlerInput {
    #[serde(rename = "realMasterState")]
    pub real_master_state: Value,
    #[serde(
        rename = "assumedMasterState",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub assumed_master_state: Option<Value>,
    #[serde(rename = "newDocumentState")]
    pub new_document_state: Value,
}

// ref: rxdb/src/types/replication-protocol.d.ts RxConflictHandler<RxDocType>
#[async_trait]
pub trait RxConflictHandler: Send + Sync {
    async fn is_equal(&self, a: &Value, b: &Value, ctx: &str) -> bool;
    async fn resolve(&self, input: &RxConflictHandlerInput, ctx: &str) -> Value;
}

// ref: rxdb/src/types/util.d.ts ById<T>
pub type ById<T> = HashMap<String, T>;

// ref: rxdb/src/types/replication-protocol.d.ts RxStorageReplicationMeta<RxDocType, CheckpointType>
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RxStorageReplicationMeta {
    pub id: String,
    #[serde(rename = "isCheckpoint")]
    pub is_checkpoint: String,
    #[serde(rename = "itemId")]
    pub item_id: String,
    #[serde(
        rename = "checkpointData",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub checkpoint_data: Option<Value>,
    #[serde(rename = "docData", default, skip_serializing_if = "Option::is_none")]
    pub doc_data: Option<Value>,
    #[serde(
        rename = "isResolvedConflict",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub is_resolved_conflict: Option<String>,
}

// ref: rxdb/src/types/replication-protocol.d.ts ReplicationActiveSubjects
pub struct ActiveSubjects {
    pub down: RxBehaviorSubject<bool>,
    pub up: RxBehaviorSubject<bool>,
    down_count: Mutex<usize>,
    up_count: Mutex<usize>,
}

impl ActiveSubjects {
    /// Marks work as pending before it can wait for a stream-queue lock.
    /// The returned guard publishes inactivity only after all such work ended.
    pub fn track(&self, direction: RxStorageReplicationDirection) -> ReplicationActivityGuard<'_> {
        let (subject, count) = self.parts(direction);
        let mut count = count.lock();
        if *count == 0 {
            subject.next(true);
        }
        *count += 1;
        drop(count);
        ReplicationActivityGuard {
            active: self,
            direction,
        }
    }

    /// Ends the initial activity represented by the subjects' initial `true` value.
    pub fn finish_initial(&self, direction: RxStorageReplicationDirection) {
        self.finish(direction);
    }

    fn finish(&self, direction: RxStorageReplicationDirection) {
        let (subject, count) = self.parts(direction);
        let mut count = count.lock();
        if *count == 0 {
            return;
        }
        *count -= 1;
        if *count == 0 {
            subject.next(false);
        }
    }

    fn parts(
        &self,
        direction: RxStorageReplicationDirection,
    ) -> (&RxBehaviorSubject<bool>, &Mutex<usize>) {
        match direction {
            RxStorageReplicationDirection::Down => (&self.down, &self.down_count),
            RxStorageReplicationDirection::Up => (&self.up, &self.up_count),
        }
    }
}

pub struct ReplicationActivityGuard<'a> {
    active: &'a ActiveSubjects,
    direction: RxStorageReplicationDirection,
}

impl Drop for ReplicationActivityGuard<'_> {
    fn drop(&mut self) {
        self.active.finish(self.direction);
    }
}

impl Default for ActiveSubjects {
    fn default() -> Self {
        Self {
            down: RxBehaviorSubject::new(true),
            up: RxBehaviorSubject::new(true),
            down_count: Mutex::new(1),
            up_count: Mutex::new(1),
        }
    }
}

// ref: rxdb/src/types/replication-protocol.d.ts ReplicationProcessedSubjects
pub struct ProcessedSubjects {
    pub down: RxSubject<Value>,
    pub up: RxSubject<Value>,
}

impl Default for ProcessedSubjects {
    fn default() -> Self {
        Self {
            down: RxSubject::new(),
            up: RxSubject::new(),
        }
    }
}

// ref: rxdb/src/types/replication-protocol.d.ts ReplicationEvents
pub struct ReplicationEvents {
    pub canceled: RxBehaviorSubject<bool>,
    pub paused: RxBehaviorSubject<bool>,
    pub active: ActiveSubjects,
    pub processed: ProcessedSubjects,
    pub resolved_conflicts: RxSubject<Value>,
    pub error: RxSubject<Value>,
}

impl ReplicationEvents {
    pub fn new() -> Self {
        Self {
            canceled: RxBehaviorSubject::new(false),
            paused: RxBehaviorSubject::new(false),
            active: ActiveSubjects::default(),
            processed: ProcessedSubjects::default(),
            resolved_conflicts: RxSubject::new(),
            error: RxSubject::new(),
        }
    }
}

impl Default for ReplicationEvents {
    fn default() -> Self {
        Self::new()
    }
}

// ref: rxdb/src/types/replication-protocol.d.ts ReplicationStats
#[derive(Debug, Default, Clone)]
pub struct DownstreamStats {
    pub add_new_task: u64,
    pub downstream_process_changes: u64,
    pub downstream_resync_once: u64,
    pub master_change_stream_emit: u64,
    pub persist_from_master: u64,
}

#[derive(Debug, Default, Clone)]
pub struct UpstreamStats {
    pub fork_change_stream_emit: u64,
    pub persist_to_master: u64,
    pub persist_to_master_conflict_writes: u64,
    pub persist_to_master_had_conflicts: u64,
    pub process_tasks: u64,
    pub upstream_initial_sync: u64,
}

#[derive(Default)]
pub struct ReplicationStats {
    pub down: parking_lot::Mutex<DownstreamStats>,
    pub up: parking_lot::Mutex<UpstreamStats>,
}

impl std::fmt::Debug for ReplicationStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReplicationStats")
            .field("down", &*self.down.lock())
            .field("up", &*self.up.lock())
            .finish()
    }
}

impl ReplicationStats {
    pub fn new() -> Self {
        Self {
            down: Mutex::new(DownstreamStats::default()),
            up: Mutex::new(UpstreamStats::default()),
        }
    }
}

// ref: rxdb/src/types/replication-protocol.d.ts FirstSyncDone
pub struct FirstSyncDone {
    pub down: RxBehaviorSubject<bool>,
    pub up: RxBehaviorSubject<bool>,
}

impl Default for FirstSyncDone {
    fn default() -> Self {
        Self {
            down: RxBehaviorSubject::new(false),
            up: RxBehaviorSubject::new(false),
        }
    }
}

// ref: rxdb/src/types/replication-protocol.d.ts StreamQueue
//
// Upstream serializes stream-processing tasks per-direction via Promise chains
// (`state.streamQueue.up = state.streamQueue.up.then(...)`). Rust mirrors that
// with one tracked `tokio::sync::Mutex` per direction. Tracking starts before
// lock acquisition, so idle waiters can observe both queued and running work.
pub struct ReplicationQueue {
    mutex: TokioMutex<()>,
    pending: Mutex<usize>,
    idle: RxBehaviorSubject<bool>,
}

impl ReplicationQueue {
    pub fn new() -> Self {
        Self {
            mutex: TokioMutex::new(()),
            pending: Mutex::new(0),
            idle: RxBehaviorSubject::new(true),
        }
    }

    pub async fn lock(&self) -> ReplicationQueueGuard<'_> {
        let entry = ReplicationQueueEntry::new(self);
        let guard = self.mutex.lock().await;
        ReplicationQueueGuard {
            guard: Some(guard),
            entry: Some(entry),
        }
    }

    pub fn is_idle(&self) -> bool {
        *self.pending.lock() == 0
    }

    pub async fn wait_until_idle(&self) {
        if self.is_idle() {
            return;
        }
        let mut idle = self.idle.subscribe();
        while let Some(is_idle) = tokio_stream::StreamExt::next(&mut idle).await {
            if is_idle && self.is_idle() {
                return;
            }
        }
    }

    fn enter(&self) {
        let mut pending = self.pending.lock();
        if *pending == 0 {
            self.idle.next(false);
        }
        *pending += 1;
    }

    fn leave(&self) {
        let mut pending = self.pending.lock();
        debug_assert!(*pending > 0);
        *pending = pending.saturating_sub(1);
        if *pending == 0 {
            self.idle.next(true);
        }
    }
}

impl Default for ReplicationQueue {
    fn default() -> Self {
        Self::new()
    }
}

struct ReplicationQueueEntry<'a> {
    queue: &'a ReplicationQueue,
}

impl<'a> ReplicationQueueEntry<'a> {
    fn new(queue: &'a ReplicationQueue) -> Self {
        queue.enter();
        Self { queue }
    }
}

impl Drop for ReplicationQueueEntry<'_> {
    fn drop(&mut self) {
        self.queue.leave();
    }
}

pub struct ReplicationQueueGuard<'a> {
    guard: Option<TokioMutexGuard<'a, ()>>,
    entry: Option<ReplicationQueueEntry<'a>>,
}

impl Drop for ReplicationQueueGuard<'_> {
    fn drop(&mut self) {
        // Publish queue-idle only after releasing the serialization lock.
        self.guard.take();
        self.entry.take();
    }
}

pub struct StreamQueue {
    pub down: ReplicationQueue,
    pub up: ReplicationQueue,
}

impl Default for StreamQueue {
    fn default() -> Self {
        Self {
            down: ReplicationQueue::new(),
            up: ReplicationQueue::new(),
        }
    }
}

// ref: rxdb/src/types/replication-protocol.d.ts RxReplicationWriteToMasterRow<RxDocType>
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RxReplicationWriteToMasterRow {
    #[serde(rename = "newDocumentState")]
    pub new_document_state: Value,
    #[serde(
        rename = "assumedMasterState",
        default,
        skip_serializing_if = "Option::is_none"
    )]
    pub assumed_master_state: Option<Value>,
}

// ref: rxdb/src/types/replication-protocol.d.ts DocumentsWithCheckpoint<RxDocType, CheckpointType>
#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
pub struct DocumentsWithCheckpoint {
    pub documents: Vec<Value>,
    pub checkpoint: Value,
}

/// Item emitted by `RxReplicationHandler::master_change_stream`.
///
/// Upstream RxDB emits either a normal `{ documents, checkpoint }` batch or the
/// string sentinel `"RESYNC"`. A Rust enum keeps that shape typed while the
/// manual serde impl preserves the browser/WebRTC wire format exactly.
#[derive(Debug, Clone, PartialEq)]
pub enum RxReplicationMasterChange {
    Documents(DocumentsWithCheckpoint),
    Resync,
}

impl From<DocumentsWithCheckpoint> for RxReplicationMasterChange {
    fn from(value: DocumentsWithCheckpoint) -> Self {
        Self::Documents(value)
    }
}

impl Serialize for RxReplicationMasterChange {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Documents(documents) => documents.serialize(serializer),
            Self::Resync => serializer.serialize_str("RESYNC"),
        }
    }
}

impl<'de> Deserialize<'de> for RxReplicationMasterChange {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        if value.as_str() == Some("RESYNC") {
            return Ok(Self::Resync);
        }
        serde_json::from_value(value)
            .map(Self::Documents)
            .map_err(serde::de::Error::custom)
    }
}

// ref: rxdb/src/types/replication-protocol.d.ts RxReplicationHandler<RxDocType, MasterCheckpointType>
#[async_trait]
pub trait RxReplicationHandler: Send + Sync {
    /// Stream of master-side change-event batches or `"RESYNC"` sentinels.
    fn master_change_stream(&self) -> RxStream<RxReplicationMasterChange>;
    /// Fetch master-side changes since the given checkpoint.
    async fn master_changes_since(
        &self,
        checkpoint: Option<Value>,
        batch_size: u64,
    ) -> Result<DocumentsWithCheckpoint, RxError>;
    /// Write rows to the master. Returns the conflicting master states (if any).
    async fn master_write(
        &self,
        rows: Vec<RxReplicationWriteToMasterRow>,
    ) -> Result<Vec<Value>, RxError>;
}

// ref: rxdb/src/types/replication-protocol.d.ts InitialCheckpoint
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct InitialCheckpoint {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub upstream: Option<Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub downstream: Option<Value>,
}

/// Closure type for the optional `waitBeforePersist` upstream throttle.
pub type WaitBeforePersist =
    Arc<dyn Fn() -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>> + Send + Sync>;

// ref: rxdb/src/types/replication-protocol.d.ts RxStorageInstanceReplicationInput<RxDocType>
pub struct RxStorageInstanceReplicationInput {
    pub identifier: String,
    pub fork_instance: Arc<dyn RxStorageInstance>,
    pub meta_instance: Arc<dyn RxStorageInstance>,
    pub hash_function: SharedHashFunction,
    pub conflict_handler: Arc<dyn RxConflictHandler>,
    pub replication_handler: Arc<dyn RxReplicationHandler>,
    pub push_batch_size: u64,
    pub pull_batch_size: u64,
    pub bulk_size: u64,
    pub keep_meta: bool,
    pub initial_checkpoint: Option<InitialCheckpoint>,
    pub wait_before_persist: Option<WaitBeforePersist>,
}

// ref: rxdb/src/types/replication-protocol.d.ts RxStorageInstanceReplicationState<RxDocType>
pub struct RxStorageInstanceReplicationState {
    pub primary_path: String,
    pub input: Arc<RxStorageInstanceReplicationInput>,
    pub checkpoint_key: String,
    pub downstream_bulk_write_flag: String,
    pub last_checkpoint_doc: Mutex<HashMap<RxStorageReplicationDirection, Value>>,
    pub events: ReplicationEvents,
    pub stats: ReplicationStats,
    pub first_sync_done: FirstSyncDone,
    pub stream_queue: StreamQueue,
    pub checkpoint_queue: TokioMutex<()>,
    pub has_attachments: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn master_change_stream_item_preserves_upstream_wire_shape() {
        let documents = RxReplicationMasterChange::Documents(DocumentsWithCheckpoint {
            documents: vec![json!({ "id": "a" })],
            checkpoint: json!({ "sequence": 1 }),
        });
        assert_eq!(
            serde_json::to_value(&documents).unwrap(),
            json!({
                "documents": [{ "id": "a" }],
                "checkpoint": { "sequence": 1 }
            })
        );
        assert_eq!(
            serde_json::from_value::<RxReplicationMasterChange>(json!("RESYNC")).unwrap(),
            RxReplicationMasterChange::Resync
        );
        assert_eq!(
            serde_json::to_value(RxReplicationMasterChange::Resync).unwrap(),
            json!("RESYNC")
        );
    }
}

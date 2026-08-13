//! Periodic tombstone cleanup for the Business OS store.
//!
//! RxDB deletes softly: a removed document stays in the table with
//! `deleted = 1` so every peer, including one that was offline while the
//! deletion happened, still learns that it is gone. The tombstone is therefore
//! load-bearing — it must outlive the slowest peer's absence.
//!
//! It must not outlive it *forever*, and until now it did. The storage layer
//! has always been able to purge tombstones (`cleanup_deleted_documents`), and
//! `RxCollection::cleanup` has always exposed it, but **nothing in the daemon
//! ever called either**. Measured on a customer instance on 13.08.2026:
//! 302.000 tombstones in `business-os-rxdb.sqlite3`, the oldest a month old,
//! 301.449 of them in `browser_frames` alone — a table whose live row count was
//! 1. Every collection accumulates them; `browser_frames` merely writes fastest.
//!
//! This is the same defect shape the sync engine has produced repeatedly: the
//! mechanism exists, is correct, and is simply never invoked on the live path.
//!
//! The retention window is deliberately generous. A tombstone dropped before a
//! peer has seen the deletion lets that peer resurrect the document on its next
//! push, which is worse than keeping the row.

use std::sync::Arc;

use rxdb::rx_database::RxDatabase;
use tokio::sync::Mutex;

/// Keep tombstones for a week before purging them.
///
/// A peer offline longer than this can resurrect a deleted document, so the
/// window is a correctness parameter, not a housekeeping preference. One week
/// matches upstream RxDB's own cleanup default.
// ref: rxdb/src/plugins/cleanup/cleanup-helper.ts (DEFAULT_CLEANUP_POLICY.minimumDeletedTime)
pub(super) const TOMBSTONE_RETENTION_MS: i64 = 7 * 24 * 60 * 60 * 1_000;

/// How often the sweep runs when it found nothing left to do.
pub(super) const TOMBSTONE_SWEEP_IDLE_INTERVAL_SECS: u64 = 30 * 60;

/// How long to wait between batches while a collection is still draining.
///
/// The first sweep on a neglected database has six figures of backlog. Pausing
/// between batches keeps the write lock available to replication instead of
/// monopolising it until the backlog is gone.
pub(super) const TOMBSTONE_SWEEP_DRAIN_INTERVAL_SECS: u64 = 5;

/// Sweep every registered collection once. Returns `true` when at least one
/// collection still has tombstones to purge, i.e. the caller should come back
/// promptly rather than sleeping the full idle interval.
pub(super) async fn sweep_tombstones_once(
    database: &Arc<RxDatabase>,
    write_lock: &Arc<Mutex<()>>,
) -> bool {
    let mut more_remains = false;
    for name in database.collection_names() {
        let Some(collection) = database.collection(&name) else {
            continue;
        };
        let _guard = write_lock.lock().await;
        match collection.cleanup(Some(TOMBSTONE_RETENTION_MS)).await {
            Ok(done) => {
                if !done {
                    more_remains = true;
                }
            }
            Err(err) => {
                // A failing collection must not stop the others, and it must not
                // silence itself either: a sweep that gives up quietly is how the
                // backlog got here.
                eprintln!("[business-os] tombstone cleanup failed for {name}: {err}");
            }
        }
    }
    more_remains
}

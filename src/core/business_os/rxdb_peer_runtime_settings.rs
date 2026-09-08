// Origin: CTOX
// License: Apache-2.0

use super::budget::PeerProjectionBudget;
use super::{
    business_os_projection_sleep_secs, record_native_peer_loop_result,
    runtime_settings_projection_stamp, update_projection_idle_rounds,
    RUNTIME_SETTINGS_PROJECTION_LOOP,
};
use crate::business_os::rxdb_peer::{project_runtime_settings_document, NativePeerLoopMetrics};
use crate::business_os::store;
use anyhow::Context;
use rxdb::rx_database::RxDatabase;
use serde_json::Value;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

pub(in crate::business_os) static SOURCE_METRICS: NativePeerLoopMetrics =
    NativePeerLoopMetrics::new("runtime_settings_source");
pub(in crate::business_os) static SOURCE_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);

struct SourceFlight(&'static AtomicU64);

impl Drop for SourceFlight {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::Relaxed);
    }
}

/// A source job can read its own SQLite connections and probe providers, but
/// cannot access RxDB or hold its writer. Only the controller publishes a ready
/// result. Dropping it aborts queued jobs; already-running blocking reads may
/// finish, but their result can no longer be projected by an obsolete peer.
struct DeferredSource<Stamp, Document> {
    stamp: Option<Stamp>,
    handle: tokio::task::JoinHandle<anyhow::Result<Document>>,
}

impl<Stamp, Document: Send + 'static> DeferredSource<Stamp, Document> {
    fn start(
        stamp: Stamp,
        build: impl FnOnce() -> anyhow::Result<Document> + Send + 'static,
        metrics: &'static NativePeerLoopMetrics,
        in_flight: &'static AtomicU64,
    ) -> Self {
        in_flight.fetch_add(1, Ordering::Relaxed);
        let flight = SourceFlight(in_flight);
        let started = Instant::now();
        let handle = tokio::task::spawn_blocking(move || {
            let _flight = flight;
            let result = build();
            metrics.record(result.as_ref().ok().map(|_| 1), started.elapsed());
            result
        });
        Self {
            stamp: Some(stamp),
            handle,
        }
    }

    fn is_ready(&self) -> bool {
        self.handle.is_finished()
    }

    async fn finish(mut self) -> anyhow::Result<(Stamp, Document)> {
        let document = (&mut self.handle)
            .await
            .context("join runtime settings source job")??;
        Ok((
            self.stamp.take().expect("source stamp is consumed once"),
            document,
        ))
    }
}

impl<Stamp, Document> Drop for DeferredSource<Stamp, Document> {
    fn drop(&mut self) {
        self.handle.abort();
    }
}

pub(super) async fn run(root: PathBuf, database: Arc<RxDatabase>) {
    let config = RUNTIME_SETTINGS_PROJECTION_LOOP;
    let mut last_source_stamp = None;
    let mut source: Option<DeferredSource<store::RuntimeSettingsProjectionStamp, Value>> = None;
    let mut consecutive_idle_rounds = 0u32;
    loop {
        let started = Instant::now();
        let result: anyhow::Result<usize> = async {
            if source.as_ref().is_some_and(DeferredSource::is_ready) {
                let (stamp, document) =
                    source.take().expect("ready source exists").finish().await?;
                let count = project_runtime_settings_document(&database, document).await?;
                // Commit the stamp captured BEFORE source loading, only after
                // successful publication. A config change during a slow probe
                // therefore schedules another job on the next tick.
                last_source_stamp = Some(stamp);
                return Ok(count);
            }
            if source.is_some() {
                return Ok(0);
            }
            let stamp = runtime_settings_projection_stamp(&root).await?;
            if last_source_stamp.as_ref() == Some(&stamp) {
                return Ok(0);
            }
            let source_root = root.clone();
            source = Some(DeferredSource::start(
                stamp,
                move || store::runtime_settings_for_rxdb(&source_root),
                &SOURCE_METRICS,
                &SOURCE_IN_FLIGHT,
            ));
            Ok(0)
        }
        .await;
        record_native_peer_loop_result(config.metrics, &result, started.elapsed());
        let sleep_for = if source.is_some() {
            // Pending work is visible in source_jobs; it must neither block a
            // peer tick nor trigger the idle/backoff path before publication.
            consecutive_idle_rounds = 0;
            PeerProjectionBudget::DEFAULT.source_poll_interval
        } else {
            update_projection_idle_rounds(
                result,
                &mut consecutive_idle_rounds,
                config.failure_prefix,
            );
            Duration::from_secs(business_os_projection_sleep_secs(
                config.active_interval_secs,
                consecutive_idle_rounds,
            ))
        };
        tokio::time::sleep(sleep_for).await;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn peer_startup_slow_source_does_not_block_polling_or_lose_its_captured_stamp() {
        static METRICS: NativePeerLoopMetrics = NativePeerLoopMetrics::new("slow_test_source");
        static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let job = DeferredSource::start(
            7u64,
            move || {
                let _ = entered_tx.send(());
                release_rx.recv_timeout(Duration::from_secs(5))?;
                Ok("snapshot for version 7")
            },
            &METRICS,
            &IN_FLIGHT,
        );
        entered_rx.await.unwrap();
        let foreground = tokio::time::timeout(Duration::from_millis(100), async {
            assert!(!job.is_ready());
            tokio::task::yield_now().await;
            assert!(!job.is_ready());
        })
        .await;
        assert!(foreground.is_ok(), "polling must not wait for the source");
        assert_eq!(IN_FLIGHT.load(Ordering::Relaxed), 1);
        release_tx.send(()).unwrap();
        let (stamp, document) = job.finish().await.unwrap();
        assert_eq!(
            stamp, 7,
            "never replace the captured stamp with a later config"
        );
        assert_eq!(document, "snapshot for version 7");
        assert_eq!(IN_FLIGHT.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn peer_startup_dropped_source_releases_its_flight_after_read_completion() {
        static METRICS: NativePeerLoopMetrics = NativePeerLoopMetrics::new("dropped_test_source");
        static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let job = DeferredSource::start(
            9,
            move || {
                let _ = entered_tx.send(());
                release_rx.recv_timeout(Duration::from_secs(5))?;
                Ok(Value::Null)
            },
            &METRICS,
            &IN_FLIGHT,
        );
        entered_rx.await.unwrap();
        drop(job);
        release_tx.send(()).unwrap();
        tokio::time::timeout(Duration::from_secs(2), async {
            while IN_FLIGHT.load(Ordering::Relaxed) != 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("a completed read must not leave a permanent in-flight flag");
    }
}

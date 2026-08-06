// ref: internal/api/server_middleware.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::middleware::response_writer::ResponseLoggingOutcome;
use crate::internal::logging::request_logger::{FileRequestLogger, RequestLogger};
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RequestLoggingMetricsSnapshot {
    pub finalized_logs: u64,
    pub forced_error_logs: u64,
    pub streaming_logs: u64,
    pub dropped_stream_chunks: u64,
    pub logger_failures: u64,
}

impl RequestLoggingMetricsSnapshot {
    fn merge(&mut self, other: Self) {
        self.finalized_logs = self.finalized_logs.saturating_add(other.finalized_logs);
        self.forced_error_logs = self
            .forced_error_logs
            .saturating_add(other.forced_error_logs);
        self.streaming_logs = self.streaming_logs.saturating_add(other.streaming_logs);
        self.dropped_stream_chunks = self
            .dropped_stream_chunks
            .saturating_add(other.dropped_stream_chunks);
        self.logger_failures = self.logger_failures.saturating_add(other.logger_failures);
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct RequestLoggingRootSnapshot {
    pub scopes: BTreeMap<String, RequestLoggingMetricsSnapshot>,
    pub total: RequestLoggingMetricsSnapshot,
}

#[derive(Debug, Default)]
pub struct RequestLoggingMetrics {
    finalized_logs: AtomicU64,
    forced_error_logs: AtomicU64,
    streaming_logs: AtomicU64,
    dropped_stream_chunks: AtomicU64,
    logger_failures: AtomicU64,
}

impl RequestLoggingMetrics {
    pub fn snapshot(&self) -> RequestLoggingMetricsSnapshot {
        RequestLoggingMetricsSnapshot {
            finalized_logs: self.finalized_logs.load(Ordering::Relaxed),
            forced_error_logs: self.forced_error_logs.load(Ordering::Relaxed),
            streaming_logs: self.streaming_logs.load(Ordering::Relaxed),
            dropped_stream_chunks: self.dropped_stream_chunks.load(Ordering::Relaxed),
            logger_failures: self.logger_failures.load(Ordering::Relaxed),
        }
    }

    pub(crate) fn record(&self, outcome: &io::Result<ResponseLoggingOutcome>) {
        match outcome {
            Ok(outcome) => {
                if outcome.log_path.is_some() {
                    self.finalized_logs.fetch_add(1, Ordering::Relaxed);
                }
                if outcome.forced_error_log {
                    self.forced_error_logs.fetch_add(1, Ordering::Relaxed);
                }
                if outcome.streaming && outcome.log_path.is_some() {
                    self.streaming_logs.fetch_add(1, Ordering::Relaxed);
                }
                self.dropped_stream_chunks
                    .fetch_add(outcome.dropped_stream_chunks, Ordering::Relaxed);
            }
            Err(_) => {
                self.logger_failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[derive(Clone)]
pub struct RequestLoggingPolicy {
    logger: Arc<dyn RequestLogger>,
    log_on_error_only: bool,
    metrics: Arc<RequestLoggingMetrics>,
}

#[derive(Debug, Default)]
pub struct RequestLoggingMetricsRegistry {
    metrics: Mutex<MetricsRegistry>,
}

impl RequestLoggingMetricsRegistry {
    pub fn snapshot_for_root(&self, root: &Path) -> RequestLoggingRootSnapshot {
        let mut registry = self
            .metrics
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        let mut snapshot = RequestLoggingRootSnapshot::default();
        registry.retain(|(registered_root, scope), metrics| {
            let Some(metrics) = metrics.upgrade() else {
                return false;
            };
            if registered_root == root {
                let metrics = metrics.snapshot();
                snapshot.total.merge(metrics);
                snapshot.scopes.insert(scope.clone(), metrics);
            }
            true
        });
        snapshot
    }

    fn register(&self, root: &Path, scope: String, metrics: &Arc<RequestLoggingMetrics>) {
        let mut registry = self
            .metrics
            .lock()
            .unwrap_or_else(|poison| poison.into_inner());
        registry.retain(|_, metrics| metrics.strong_count() > 0);
        registry.insert((root.to_path_buf(), scope), Arc::downgrade(metrics));
    }
}

impl RequestLoggingPolicy {
    pub fn error_only(logs_dir: impl AsRef<Path>, error_logs_max_files: usize) -> Self {
        Self::new(Arc::new(FileRequestLogger::new(
            false,
            logs_dir,
            "",
            error_logs_max_files,
        )))
    }

    pub fn full(logs_dir: impl AsRef<Path>, error_logs_max_files: usize) -> Self {
        Self::new(Arc::new(FileRequestLogger::new(
            true,
            logs_dir,
            "",
            error_logs_max_files,
        )))
    }

    pub fn error_only_scoped(
        registry: &RequestLoggingMetricsRegistry,
        root: impl AsRef<Path>,
        scope: impl Into<String>,
        logs_dir: impl AsRef<Path>,
        error_logs_max_files: usize,
    ) -> Self {
        let policy = Self::error_only(logs_dir, error_logs_max_files);
        registry.register(root.as_ref(), scope.into(), &policy.metrics);
        policy
    }

    pub fn full_scoped(
        registry: &RequestLoggingMetricsRegistry,
        root: impl AsRef<Path>,
        scope: impl Into<String>,
        logs_dir: impl AsRef<Path>,
        error_logs_max_files: usize,
    ) -> Self {
        let policy = Self::full(logs_dir, error_logs_max_files);
        registry.register(root.as_ref(), scope.into(), &policy.metrics);
        policy
    }

    pub fn new(logger: Arc<dyn RequestLogger>) -> Self {
        let log_on_error_only = !logger.is_enabled();
        Self {
            logger,
            log_on_error_only,
            metrics: Arc::new(RequestLoggingMetrics::default()),
        }
    }

    pub(crate) fn logger(&self) -> Arc<dyn RequestLogger> {
        Arc::clone(&self.logger)
    }

    pub(crate) fn log_on_error_only(&self) -> bool {
        self.log_on_error_only
    }

    pub fn metrics(&self) -> Arc<RequestLoggingMetrics> {
        Arc::clone(&self.metrics)
    }
}

type MetricsRegistry = BTreeMap<(PathBuf, String), Weak<RequestLoggingMetrics>>;

// ref: sdk/cliproxy/auth/auto_refresh_loop.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::{DateTime, Utc};
use tokio::sync::{mpsc, watch, Notify};
use tokio::task::JoinHandle;

use super::conductor_refresh::{
    has_unauthorized_auth_failure, last_refresh_time, preferred_refresh_interval,
    REFRESH_CHECK_INTERVAL, REFRESH_MAX_CONCURRENCY, REFRESH_PENDING_BACKOFF,
};
use super::types::is_go_zero_time;
use super::{
    should_refresh, Auth, AuthKind, AuthLifecycle, AuthRefresher, ModelResumeSink,
    RefreshCancellation, RefreshTransactionError,
};

#[must_use]
pub fn next_refresh_check_at(
    now: DateTime<Utc>,
    auth: &Auth,
    interval: Duration,
) -> Option<DateTime<Utc>> {
    if has_unauthorized_auth_failure(auth) || auth.auth_kind() == Some(AuthKind::ApiKey) {
        return None;
    }
    if !is_go_zero_time(&auth.next_refresh_after) && now < auth.next_refresh_after {
        return Some(auth.next_refresh_after);
    }
    if auth
        .runtime
        .as_deref()
        .is_some_and(|runtime| runtime.evaluates_refresh())
    {
        return add_duration(now, nonzero_or_default(interval));
    }

    let last_refresh = last_refresh_time(auth);
    let expiration = auth
        .expiration_time()
        .filter(|expiration| !is_go_zero_time(expiration));

    if let Some(preferred) = preferred_refresh_interval(auth) {
        if let Some(expiration) = expiration {
            if expiration <= now || within(expiration, now, preferred) {
                return Some(now);
            }
        }
        let Some(last_refresh) = last_refresh else {
            return Some(now);
        };
        let mut candidates = Vec::with_capacity(2);
        if let Some(expiration) = expiration {
            candidates.push(sub_duration(expiration, preferred)?);
        }
        candidates.push(add_duration(last_refresh, preferred)?);
        let next = candidates.into_iter().min()?;
        return Some(next.max(now));
    }

    let lead = auth.refresh_lead()?;
    if let Some(expiration) = expiration {
        return sub_duration(expiration, lead).map(|due| due.max(now));
    }
    if let Some(last_refresh) = last_refresh {
        return add_duration(last_refresh, lead).map(|due| due.max(now));
    }
    Some(now)
}

fn nonzero_or_default(interval: Duration) -> Duration {
    if interval.is_zero() {
        REFRESH_CHECK_INTERVAL
    } else {
        interval
    }
}

fn add_duration(timestamp: DateTime<Utc>, duration: Duration) -> Option<DateTime<Utc>> {
    chrono::Duration::from_std(duration)
        .ok()
        .and_then(|duration| timestamp.checked_add_signed(duration))
}

fn sub_duration(timestamp: DateTime<Utc>, duration: Duration) -> Option<DateTime<Utc>> {
    chrono::Duration::from_std(duration)
        .ok()
        .and_then(|duration| timestamp.checked_sub_signed(duration))
}

fn within(later: DateTime<Utc>, earlier: DateTime<Utc>, duration: Duration) -> bool {
    later
        .signed_duration_since(earlier)
        .to_std()
        .is_ok_and(|remaining| remaining <= duration)
}

/// Concurrent, updateable due-time index replacing Go's mutex-protected heap.
/// The ordered pair additionally makes equal-deadline dispatch deterministic.
#[derive(Default)]
pub struct RefreshSchedule {
    state: Mutex<RefreshScheduleState>,
    wake: Notify,
}

#[derive(Default)]
struct RefreshScheduleState {
    by_due: BTreeSet<(DateTime<Utc>, String)>,
    by_id: BTreeMap<String, DateTime<Utc>>,
}

impl RefreshSchedule {
    pub fn upsert(&self, auth_id: &str, due_at: DateTime<Utc>) -> bool {
        let auth_id = auth_id.trim();
        if auth_id.is_empty() || is_go_zero_time(&due_at) {
            return false;
        }
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(previous) = state.by_id.insert(auth_id.to_owned(), due_at) {
            state.by_due.remove(&(previous, auth_id.to_owned()));
        }
        state.by_due.insert((due_at, auth_id.to_owned()));
        drop(state);
        self.wake.notify_one();
        true
    }

    pub fn remove(&self, auth_id: &str) -> bool {
        let auth_id = auth_id.trim();
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(previous) = state.by_id.remove(auth_id) else {
            return false;
        };
        let removed = state.by_due.remove(&(previous, auth_id.to_owned()));
        drop(state);
        self.wake.notify_one();
        removed
    }

    #[must_use]
    pub fn peek(&self) -> Option<DateTime<Utc>> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .by_due
            .first()
            .map(|(due, _)| *due)
    }

    #[must_use]
    pub fn pop_due(&self, now: DateTime<Utc>) -> Vec<String> {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let due = state
            .by_due
            .iter()
            .take_while(|(due, _)| *due <= now)
            .cloned()
            .collect::<Vec<_>>();
        for (due_at, auth_id) in &due {
            state.by_due.remove(&(*due_at, auth_id.clone()));
            state.by_id.remove(auth_id);
        }
        due.into_iter().map(|(_, auth_id)| auth_id).collect()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .by_id
            .len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub async fn notified(&self) {
        self.wake.notified().await;
    }
}

impl std::fmt::Debug for RefreshSchedule {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RefreshSchedule")
            .field("entries", &self.len())
            .finish()
    }
}

pub trait AutoRefreshClock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

#[derive(Debug, Default)]
pub struct SystemAutoRefreshClock;

impl AutoRefreshClock for SystemAutoRefreshClock {
    fn now(&self) -> DateTime<Utc> {
        std::time::SystemTime::now().into()
    }
}

pub trait AuthRefresherResolver: Send + Sync {
    fn resolve(&self, provider: &str) -> Option<Arc<dyn AuthRefresher>>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AutoRefreshConfig {
    pub interval: Duration,
    pub concurrency: usize,
    pub job_buffer: usize,
}

impl Default for AutoRefreshConfig {
    fn default() -> Self {
        let concurrency = REFRESH_MAX_CONCURRENCY;
        Self {
            interval: REFRESH_CHECK_INTERVAL,
            concurrency,
            job_buffer: (concurrency * 4).max(64),
        }
    }
}

impl AutoRefreshConfig {
    #[must_use]
    pub fn normalized(self) -> Self {
        let interval = if self.interval.is_zero() {
            REFRESH_CHECK_INTERVAL
        } else {
            self.interval
        };
        let concurrency = if self.concurrency == 0 {
            REFRESH_MAX_CONCURRENCY
        } else {
            self.concurrency
        };
        let job_buffer = if self.job_buffer == 0 {
            (concurrency * 4).max(64)
        } else {
            self.job_buffer.max(concurrency)
        };
        Self {
            interval,
            concurrency,
            job_buffer,
        }
    }
}

/// Owned auto-refresh runtime. Drop requests cancellation; `stop` additionally
/// waits until the dispatcher and its bounded worker set have exited.
pub struct AutoRefreshWorker {
    stop: watch::Sender<bool>,
    cancellation: RefreshCancellation,
    task: Option<JoinHandle<()>>,
    config: AutoRefreshConfig,
}

impl AutoRefreshWorker {
    #[must_use]
    pub fn spawn(
        lifecycle: Arc<AuthLifecycle>,
        schedule: Arc<RefreshSchedule>,
        resolver: Arc<dyn AuthRefresherResolver>,
        resume_sink: Arc<dyn ModelResumeSink>,
        clock: Arc<dyn AutoRefreshClock>,
        config: AutoRefreshConfig,
    ) -> Self {
        let config = config.normalized();
        let (stop, stop_rx) = watch::channel(false);
        let cancellation = RefreshCancellation::default();
        let runtime = Arc::new(RefreshWorkerRuntime {
            lifecycle,
            schedule,
            resolver,
            resume_sink,
            clock,
            interval: config.interval,
            cancellation: cancellation.clone(),
            pending: Mutex::new(BTreeSet::new()),
        });
        let task = tokio::spawn(run_auto_refresh(runtime, config, stop_rx));
        Self {
            stop,
            cancellation,
            task: Some(task),
            config,
        }
    }

    pub async fn stop(mut self) {
        self.request_stop();
        if let Some(task) = self.task.take() {
            let _ = task.await;
        }
    }

    fn request_stop(&mut self) {
        self.cancellation.cancel();
        let _ = self.stop.send(true);
    }
}

impl Drop for AutoRefreshWorker {
    fn drop(&mut self) {
        self.request_stop();
    }
}

impl fmt::Debug for AutoRefreshWorker {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AutoRefreshWorker")
            .field("config", &self.config)
            .field("running", &self.task.is_some())
            .finish()
    }
}

type SharedJobReceiver = Arc<tokio::sync::Mutex<mpsc::Receiver<String>>>;

async fn run_auto_refresh(
    runtime: Arc<RefreshWorkerRuntime>,
    config: AutoRefreshConfig,
    mut stop: watch::Receiver<bool>,
) {
    let (jobs, job_rx) = mpsc::channel(config.job_buffer);
    let job_rx = Arc::new(tokio::sync::Mutex::new(job_rx));
    let mut workers = Vec::with_capacity(config.concurrency);
    for _ in 0..config.concurrency {
        workers.push(tokio::spawn(run_refresh_worker(
            runtime.clone(),
            job_rx.clone(),
            stop.clone(),
        )));
    }

    loop {
        if *stop.borrow() {
            break;
        }
        let now = runtime.clock.now();
        if let Some(next) = runtime.schedule.peek() {
            let wait = next
                .signed_duration_since(now)
                .to_std()
                .unwrap_or(Duration::ZERO);
            if wait.is_zero() {
                let due = runtime
                    .schedule
                    .pop_due(now)
                    .into_iter()
                    .filter(|auth_id| runtime.mark_pending(auth_id))
                    .collect::<Vec<_>>();
                for auth_id in due {
                    tokio::select! {
                        changed = stop.changed() => {
                            if changed.is_err() || *stop.borrow() {
                                break;
                            }
                        }
                        sent = jobs.send(auth_id) => {
                            if sent.is_err() {
                                break;
                            }
                        }
                    }
                }
                continue;
            }
            tokio::select! {
                changed = stop.changed() => {
                    if changed.is_err() || *stop.borrow() {
                        break;
                    }
                }
                () = runtime.schedule.notified() => {}
                () = tokio::time::sleep(wait) => {}
            }
        } else {
            tokio::select! {
                changed = stop.changed() => {
                    if changed.is_err() || *stop.borrow() {
                        break;
                    }
                }
                () = runtime.schedule.notified() => {}
            }
        }
    }

    drop(jobs);
    for worker in workers {
        let _ = worker.await;
    }
    runtime.requeue_pending();
}

struct RefreshWorkerRuntime {
    lifecycle: Arc<AuthLifecycle>,
    schedule: Arc<RefreshSchedule>,
    resolver: Arc<dyn AuthRefresherResolver>,
    resume_sink: Arc<dyn ModelResumeSink>,
    clock: Arc<dyn AutoRefreshClock>,
    interval: Duration,
    cancellation: RefreshCancellation,
    pending: Mutex<BTreeSet<String>>,
}

impl RefreshWorkerRuntime {
    fn mark_pending(&self, auth_id: &str) -> bool {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(auth_id.to_owned())
    }

    fn clear_pending(&self, auth_id: &str) {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(auth_id);
    }

    fn requeue_pending(&self) {
        let pending = std::mem::take(
            &mut *self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        let now = self.clock.now();
        if let Some(retry) = add_duration(now, REFRESH_PENDING_BACKOFF) {
            for auth_id in pending {
                self.schedule.upsert(&auth_id, retry);
            }
        }
    }
}

struct PendingRefreshGuard {
    runtime: Arc<RefreshWorkerRuntime>,
    auth_id: String,
}

impl PendingRefreshGuard {
    fn new(runtime: Arc<RefreshWorkerRuntime>, auth_id: String) -> Self {
        Self { runtime, auth_id }
    }
}

impl Drop for PendingRefreshGuard {
    fn drop(&mut self) {
        self.runtime.clear_pending(&self.auth_id);
    }
}

async fn run_refresh_worker(
    runtime: Arc<RefreshWorkerRuntime>,
    jobs: SharedJobReceiver,
    mut stop: watch::Receiver<bool>,
) {
    loop {
        let auth_id = tokio::select! {
            changed = stop.changed() => {
                if changed.is_err() || *stop.borrow() {
                    return;
                }
                continue;
            }
            auth_id = receive_job(&jobs) => match auth_id {
                Some(auth_id) => auth_id,
                None => return,
            }
        };
        if *stop.borrow() {
            return;
        }
        let _pending = PendingRefreshGuard::new(runtime.clone(), auth_id.clone());
        let now = runtime.clock.now();
        let Some(auth) = runtime.lifecycle.get_cached(&auth_id) else {
            continue;
        };
        let Some(next) = next_refresh_check_at(now, &auth, runtime.interval) else {
            runtime.schedule.remove(&auth_id);
            continue;
        };
        if !should_refresh(&auth, now) {
            runtime.schedule.upsert(&auth_id, next);
            continue;
        }
        let Some(refresher) = runtime.resolver.resolve(&auth.provider) else {
            if let Some(retry) = add_duration(now, runtime.interval) {
                runtime.schedule.upsert(&auth_id, retry);
            }
            continue;
        };

        let refresh_lifecycle = runtime.lifecycle.clone();
        let refresh_sink = runtime.resume_sink.clone();
        let refresh_cancellation = runtime.cancellation.clone();
        let refresh_auth_id = auth_id.clone();
        let result = tokio::task::spawn_blocking(move || {
            refresh_lifecycle.refresh_with_cancellation(
                &refresh_auth_id,
                None,
                now,
                refresher.as_ref(),
                refresh_sink.as_ref(),
                &refresh_cancellation,
            )
        })
        .await;
        let needs_pending_backoff = match result {
            Ok(Ok(_)) => false,
            Ok(Err(super::AuthLifecycleRefreshError::Refresh(
                RefreshTransactionError::Refresh(_),
            ))) => false,
            Ok(Err(_)) | Err(_) => true,
        };
        if needs_pending_backoff {
            if let Some(retry) = add_duration(now, REFRESH_PENDING_BACKOFF) {
                runtime.schedule.upsert(&auth_id, retry);
            }
        }
    }
}

fn receive_job(
    jobs: &SharedJobReceiver,
) -> Pin<Box<dyn Future<Output = Option<String>> + Send + '_>> {
    Box::pin(async move { jobs.lock().await.recv().await })
}

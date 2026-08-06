// ref: internal/redisqueue/queue.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use tokio::sync::mpsc;

const DEFAULT_RETENTION_SECONDS: i64 = 60;
const MAX_RETENTION_SECONDS: i64 = 3_600;
const USAGE_SUBSCRIBER_BUFFER: usize = 256;
const ERROR_SUBSCRIBER_BUFFER: usize = 256;

const USAGE_SUPPORT_REFRESH_PAYLOAD: &[u8] = br#"{"support_refresh":true}"#;
const USAGE_REFRESH_PAYLOAD: &[u8] = br#"{"refresh":true}"#;

#[derive(Clone)]
struct QueueItem {
    enqueued_at: Instant,
    payload: Vec<u8>,
}

#[derive(Default)]
struct QueueState {
    items: VecDeque<QueueItem>,
    subscribers: BTreeMap<u64, mpsc::Sender<Vec<u8>>>,
    next_subscriber_id: u64,
}

/// Instance-owned port of the upstream process-global usage/error queues.
///
/// CTOX owns runtime lifecycles explicitly, so disabling one gateway queue
/// cannot clear subscribers belonging to another gateway instance.
pub struct UsageQueue {
    enabled: AtomicBool,
    retention_seconds: AtomicI64,
    usage: Arc<Mutex<QueueState>>,
    errors: Arc<Mutex<QueueState>>,
}

impl Default for UsageQueue {
    fn default() -> Self {
        Self {
            enabled: AtomicBool::new(false),
            retention_seconds: AtomicI64::new(DEFAULT_RETENTION_SECONDS),
            usage: Arc::new(Mutex::new(QueueState::default())),
            errors: Arc::new(Mutex::new(QueueState::default())),
        }
    }
}

impl UsageQueue {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Release);
        if !enabled {
            clear(&self.usage);
            clear(&self.errors);
        }
    }

    pub fn enabled(&self) -> bool {
        self.enabled.load(Ordering::Acquire)
    }

    pub fn set_retention_seconds(&self, seconds: i64) {
        let normalized = if seconds <= 0 {
            DEFAULT_RETENTION_SECONDS
        } else {
            seconds.min(MAX_RETENTION_SECONDS)
        };
        self.retention_seconds.store(normalized, Ordering::Release);
    }

    pub fn retention_seconds(&self) -> i64 {
        self.retention_seconds.load(Ordering::Acquire)
    }

    pub fn enqueue(&self, payload: &[u8]) {
        if !self.enabled() || payload.is_empty() {
            return;
        }
        if publish_to_subscribers(&self.usage, payload) {
            return;
        }
        enqueue(
            &self.usage,
            payload,
            self.retention_duration(),
            Instant::now(),
        );
    }

    pub fn enqueue_error(&self, payload: &[u8]) {
        if !self.enabled() || payload.is_empty() {
            return;
        }
        publish_to_subscribers(&self.errors, payload);
    }

    pub fn pop_oldest(&self, count: usize) -> Vec<Vec<u8>> {
        if !self.enabled() || count == 0 {
            return Vec::new();
        }
        pop_oldest(
            &self.usage,
            count,
            self.retention_duration(),
            Instant::now(),
        )
    }

    pub fn subscribe_usage(&self) -> (mpsc::Receiver<Vec<u8>>, Subscription) {
        subscribe(
            &self.usage,
            USAGE_SUBSCRIBER_BUFFER,
            Some(USAGE_SUPPORT_REFRESH_PAYLOAD),
        )
    }

    pub fn subscribe_errors(&self) -> (mpsc::Receiver<Vec<u8>>, Subscription) {
        subscribe(&self.errors, ERROR_SUBSCRIBER_BUFFER, None)
    }

    pub fn notify_usage_refresh(&self) {
        publish_to_subscribers(&self.usage, USAGE_REFRESH_PAYLOAD);
    }

    fn retention_duration(&self) -> Duration {
        Duration::from_secs(self.retention_seconds().max(1) as u64)
    }
}

/// Idempotent subscription ownership. Dropping it has the same effect as the
/// upstream unsubscribe closure.
pub struct Subscription {
    queue: Weak<Mutex<QueueState>>,
    id: Option<u64>,
}

impl Subscription {
    pub fn unsubscribe(&mut self) {
        let Some(id) = self.id.take() else { return };
        let Some(queue) = self.queue.upgrade() else {
            return;
        };
        queue
            .lock()
            .expect("usage queue poisoned")
            .subscribers
            .remove(&id);
    }
}

impl Drop for Subscription {
    fn drop(&mut self) {
        self.unsubscribe();
    }
}

fn clear(queue: &Arc<Mutex<QueueState>>) {
    let mut queue = queue.lock().expect("usage queue poisoned");
    queue.items.clear();
    queue.subscribers.clear();
}

fn enqueue(queue: &Arc<Mutex<QueueState>>, payload: &[u8], retention: Duration, now: Instant) {
    let mut queue = queue.lock().expect("usage queue poisoned");
    prune(&mut queue, retention, now);
    queue.items.push_back(QueueItem {
        enqueued_at: now,
        payload: payload.to_vec(),
    });
}

fn publish_to_subscribers(queue: &Arc<Mutex<QueueState>>, payload: &[u8]) -> bool {
    let mut queue = queue.lock().expect("usage queue poisoned");
    if queue.subscribers.is_empty() {
        return false;
    }
    queue
        .subscribers
        .retain(|_, subscriber| subscriber.try_send(payload.to_vec()).is_ok());
    true
}

fn subscribe(
    queue: &Arc<Mutex<QueueState>>,
    buffer: usize,
    initial_payload: Option<&[u8]>,
) -> (mpsc::Receiver<Vec<u8>>, Subscription) {
    let (sender, receiver) = mpsc::channel(buffer);
    if let Some(payload) = initial_payload {
        sender
            .try_send(payload.to_vec())
            .expect("initial usage payload fits the subscriber buffer");
    }
    let id = {
        let mut queue = queue.lock().expect("usage queue poisoned");
        queue.next_subscriber_id = queue.next_subscriber_id.wrapping_add(1);
        let id = queue.next_subscriber_id;
        queue.subscribers.insert(id, sender);
        id
    };
    (
        receiver,
        Subscription {
            queue: Arc::downgrade(queue),
            id: Some(id),
        },
    )
}

fn pop_oldest(
    queue: &Arc<Mutex<QueueState>>,
    count: usize,
    retention: Duration,
    now: Instant,
) -> Vec<Vec<u8>> {
    let mut queue = queue.lock().expect("usage queue poisoned");
    prune(&mut queue, retention, now);
    let count = count.min(queue.items.len());
    queue
        .items
        .drain(..count)
        .map(|item| item.payload)
        .collect()
}

fn prune(queue: &mut QueueState, retention: Duration, now: Instant) {
    while queue
        .items
        .front()
        .is_some_and(|item| now.duration_since(item.enqueued_at) > retention)
    {
        queue.items.pop_front();
    }
}

// ref: internal/tui/loghook.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::internal::logging::global_logger::{LogEntry, LogFormatter};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Mutex};

pub struct LogHook {
    capacity: usize,
    formatter: LogFormatter,
    lines: Mutex<VecDeque<String>>,
    subscribers: Mutex<Vec<mpsc::SyncSender<String>>>,
    dropped: AtomicU64,
}
impl LogHook {
    pub fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            formatter: LogFormatter,
            lines: Mutex::new(VecDeque::with_capacity(capacity.max(1))),
            subscribers: Mutex::new(Vec::new()),
            dropped: AtomicU64::new(0),
        }
    }
    pub fn fire(&self, entry: &LogEntry) {
        let line = self.formatter.format(entry);
        {
            let mut lines = self.lines.lock().unwrap_or_else(|p| p.into_inner());
            if lines.len() == self.capacity {
                lines.pop_front();
            }
            lines.push_back(line.clone());
        }
        self.subscribers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .retain(|sender| match sender.try_send(line.clone()) {
                Ok(()) => true,
                Err(mpsc::TrySendError::Full(_)) => {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(mpsc::TrySendError::Disconnected(_)) => false,
            });
    }
    pub fn subscribe(&self, capacity: usize) -> mpsc::Receiver<String> {
        let (sender, receiver) = mpsc::sync_channel(capacity.max(1));
        self.subscribers
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .push(sender);
        receiver
    }
    pub fn snapshot(&self) -> Vec<String> {
        self.lines
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .iter()
            .cloned()
            .collect()
    }
    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }
}

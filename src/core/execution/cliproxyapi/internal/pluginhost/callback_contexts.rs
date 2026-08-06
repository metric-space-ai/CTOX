// ref: internal/pluginhost/callback_contexts.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: callback authority is scoped to the isolated plugin process
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, Weak};

#[derive(Clone, Debug)]
pub struct CallbackAuthority {
    plugin_id: String,
    deadline_unix_ms: Option<u64>,
    cancelled: Arc<AtomicBool>,
}

impl CallbackAuthority {
    pub fn new(plugin_id: impl Into<String>, deadline_unix_ms: Option<u64>) -> Self {
        Self {
            plugin_id: plugin_id.into().trim().to_owned(),
            deadline_unix_ms,
            cancelled: Arc::new(AtomicBool::new(false)),
        }
    }

    pub fn plugin_id(&self) -> &str {
        &self.plugin_id
    }

    pub fn deadline_unix_ms(&self) -> Option<u64> {
        self.deadline_unix_ms
    }

    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }
}

type Cleanup = Box<dyn FnOnce() + Send + 'static>;

struct CallbackEntry {
    authority: CallbackAuthority,
    cleanup: Vec<Cleanup>,
}

struct RegistryInner {
    next: AtomicU64,
    entries: Mutex<BTreeMap<String, CallbackEntry>>,
}

#[derive(Clone)]
pub struct CallbackContextRegistry {
    inner: Arc<RegistryInner>,
}

impl Default for CallbackContextRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl CallbackContextRegistry {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(RegistryInner {
                next: AtomicU64::new(0),
                entries: Mutex::new(BTreeMap::new()),
            }),
        }
    }

    pub fn open(&self, authority: CallbackAuthority) -> CallbackContextLease {
        let id = self
            .inner
            .next
            .fetch_add(1, Ordering::Relaxed)
            .saturating_add(1)
            .to_string();
        self.inner
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .insert(
                id.clone(),
                CallbackEntry {
                    authority,
                    cleanup: Vec::new(),
                },
            );
        CallbackContextLease {
            id,
            registry: Arc::downgrade(&self.inner),
            closed: false,
        }
    }

    pub fn resolve(&self, id: &str) -> Option<CallbackAuthority> {
        self.inner
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(id)
            .map(|entry| entry.authority.clone())
    }

    pub fn plugin_id(&self, id: &str) -> Option<String> {
        self.resolve(id).map(|authority| authority.plugin_id)
    }

    pub fn add_cleanup(&self, id: &str, cleanup: Cleanup) -> bool {
        let mut entries = self
            .inner
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(entry) = entries.get_mut(id) {
            entry.cleanup.push(cleanup);
            true
        } else {
            drop(entries);
            cleanup();
            false
        }
    }

    pub fn len(&self) -> usize {
        self.inner
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub struct CallbackContextLease {
    id: String,
    registry: Weak<RegistryInner>,
    closed: bool,
}

impl CallbackContextLease {
    pub fn id(&self) -> &str {
        &self.id
    }

    pub fn close(&mut self) {
        if self.closed {
            return;
        }
        self.closed = true;
        let Some(registry) = self.registry.upgrade() else {
            return;
        };
        let entry = registry
            .entries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id);
        if let Some(entry) = entry {
            entry.authority.cancel();
            for cleanup in entry.cleanup {
                cleanup();
            }
        }
    }
}

impl Drop for CallbackContextLease {
    fn drop(&mut self) {
        self.close();
    }
}

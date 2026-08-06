// ref: internal/cache/bounded_lru.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: Rust values are cloned on read, matching Go's value-return contract.
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{HashMap, VecDeque};
use std::hash::Hash;
use std::sync::{Arc, Mutex};

type EvictionCallback<K, V> = dyn Fn(K, V) + Send + Sync + 'static;

struct State<K, V> {
    entries: HashMap<K, V>,
    order: VecDeque<K>,
}

/// A concurrency-safe, capacity-bounded least-recently-used cache.
///
/// `create` runs while the cache lock is held so concurrent misses for the
/// same key produce exactly one value. Eviction callbacks run after the lock
/// is released and may therefore safely call back into the cache.
pub struct BoundedLru<K, V> {
    capacity: usize,
    state: Mutex<State<K, V>>,
    on_evict: Option<Arc<EvictionCallback<K, V>>>,
}

impl<K, V> BoundedLru<K, V>
where
    K: Clone + Eq + Hash,
    V: Clone,
{
    #[must_use]
    pub fn new(capacity: usize, on_evict: Option<Arc<EvictionCallback<K, V>>>) -> Self {
        let capacity = capacity.max(1);
        Self {
            capacity,
            state: Mutex::new(State {
                entries: HashMap::with_capacity(capacity),
                order: VecDeque::with_capacity(capacity),
            }),
            on_evict,
        }
    }

    pub fn get_or_add(&self, key: K, create: impl FnOnce() -> V) -> V {
        let (value, evicted) = {
            let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
            if let Some(value) = state.entries.get(&key).cloned() {
                touch(&mut state.order, &key);
                return value;
            }

            let value = create();
            state.entries.insert(key.clone(), value.clone());
            state.order.push_front(key);
            let evicted = if state.order.len() > self.capacity {
                state
                    .order
                    .pop_back()
                    .and_then(|oldest| state.entries.remove(&oldest).map(|value| (oldest, value)))
            } else {
                None
            };
            (value, evicted)
        };

        if let (Some(callback), Some((key, value))) = (&self.on_evict, evicted) {
            callback(key, value);
        }
        value
    }

    pub fn get(&self, key: &K) -> Option<V> {
        let mut state = self.state.lock().unwrap_or_else(|error| error.into_inner());
        let value = state.entries.get(key).cloned()?;
        touch(&mut state.order, key);
        Some(value)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.state
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entries
            .len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

fn touch<K: Eq>(order: &mut VecDeque<K>, key: &K) {
    if let Some(index) = order.iter().position(|candidate| candidate == key) {
        if let Some(key) = order.remove(index) {
            order.push_front(key);
        }
    }
}

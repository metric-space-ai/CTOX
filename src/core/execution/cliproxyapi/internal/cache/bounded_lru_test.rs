// ref: internal/cache/bounded_lru_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread;

use super::bounded_lru::BoundedLru;

type SharedStringCache = Arc<BoundedLru<String, String>>;
type OptionalSharedStringCache = Arc<Mutex<Option<SharedStringCache>>>;

#[test]
fn evicts_least_recently_used() {
    let evicted = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&evicted);
    let cache = BoundedLru::new(
        2,
        Some(Arc::new(move |key: String, value: String| {
            sink.lock()
                .unwrap_or_else(|error| error.into_inner())
                .push(format!("{key}={value}"));
        })),
    );

    assert_eq!(cache.get_or_add("a".into(), || "A".into()), "A");
    assert_eq!(cache.get_or_add("b".into(), || "B".into()), "B");
    assert_eq!(cache.get(&"a".into()).as_deref(), Some("A"));
    assert_eq!(cache.get_or_add("c".into(), || "C".into()), "C");

    assert_eq!(cache.get(&"b".into()), None);
    assert_eq!(cache.len(), 2);
    assert_eq!(
        *evicted.lock().unwrap_or_else(|error| error.into_inner()),
        vec!["b=B"]
    );
}

#[test]
fn creates_one_value_per_key_concurrently() {
    let cache = Arc::new(BoundedLru::<String, usize>::new(2, None));
    let started = Arc::new(Barrier::new(2));
    let release = Arc::new(Barrier::new(2));
    let creates = Arc::new(AtomicUsize::new(0));

    let first_cache = Arc::clone(&cache);
    let first_started = Arc::clone(&started);
    let first_release = Arc::clone(&release);
    let first_creates = Arc::clone(&creates);
    let first = thread::spawn(move || {
        first_cache.get_or_add("key".into(), || {
            first_creates.fetch_add(1, Ordering::SeqCst);
            first_started.wait();
            first_release.wait();
            42
        })
    });

    started.wait();
    let second_cache = Arc::clone(&cache);
    let second_creates = Arc::clone(&creates);
    let second = thread::spawn(move || {
        second_cache.get_or_add("key".into(), || {
            second_creates.fetch_add(1, Ordering::SeqCst);
            7
        })
    });
    release.wait();

    assert_eq!(first.join().expect("first thread"), 42);
    assert_eq!(second.join().expect("second thread"), 42);
    assert_eq!(creates.load(Ordering::SeqCst), 1);
}

#[test]
fn eviction_callback_runs_after_unlock() {
    let holder: OptionalSharedStringCache = Arc::new(Mutex::new(None));
    let callback_holder = Arc::clone(&holder);
    let cache = Arc::new(BoundedLru::new(
        1,
        Some(Arc::new(move |_: String, _: String| {
            let cache = callback_holder
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .as_ref()
                .expect("cache installed")
                .clone();
            assert_eq!(cache.len(), 1);
        })),
    ));
    *holder.lock().unwrap_or_else(|error| error.into_inner()) = Some(Arc::clone(&cache));
    cache.get_or_add("a".into(), || "A".into());
    cache.get_or_add("b".into(), || "B".into());
}

// ref: sdk/cliproxy/executionregistry/registry_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{Registry, RegistryError, ScopeSpec, State, WaitBudget};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{mpsc, Arc, Barrier};
use std::time::{Duration, Instant};

fn short_budget() -> WaitBudget {
    WaitBudget::for_duration(Duration::from_secs(1))
}

fn wait_for_state(registry: &Registry, expected: State) {
    let deadline = Instant::now() + Duration::from_secs(1);
    while registry.state() != expected {
        assert!(
            Instant::now() < deadline,
            "registry did not reach {expected:?}"
        );
        std::thread::sleep(Duration::from_millis(1));
    }
}

#[test]
fn drain_rejects_late_install_and_cancels_bound_scopes() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry
        .install(
            &pending,
            ScopeSpec {
                request_id: "req-1".into(),
                credential_id: "cred-1".into(),
                model: "gpt".into(),
                kind: "http".into(),
                ..ScopeSpec::default()
            },
        )
        .unwrap();
    let closed = Arc::new(AtomicU32::new(0));
    let scope_for_close = scope.clone();
    let closed_for_close = Arc::clone(&closed);
    scope
        .bind(move || {
            closed_for_close.fetch_add(1, Ordering::SeqCst);
            std::thread::spawn(move || scope_for_close.end("canceled"));
            Ok(())
        })
        .unwrap();

    registry.drain(short_budget()).unwrap();
    assert_eq!(closed.load(Ordering::SeqCst), 1);
    assert!(matches!(
        registry.begin_dispatch(),
        Err(RegistryError::NotAccepting)
    ));
}

#[test]
fn scope_end_is_exactly_once() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry.install(&pending, ScopeSpec::default()).unwrap();
    let closed = Arc::new(AtomicU32::new(0));
    let closed_for_close = Arc::clone(&closed);
    scope
        .bind(move || {
            closed_for_close.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap();

    let barrier = Arc::new(Barrier::new(3));
    let first = scope.clone();
    let first_barrier = Arc::clone(&barrier);
    let first_thread = std::thread::spawn(move || {
        first_barrier.wait();
        first.end("complete");
    });
    let second = scope.clone();
    let second_barrier = Arc::clone(&barrier);
    let second_thread = std::thread::spawn(move || {
        second_barrier.wait();
        second.end("duplicate");
    });
    barrier.wait();
    first_thread.join().unwrap();
    second_thread.join().unwrap();
    assert_eq!(closed.load(Ordering::SeqCst), 1);
}

#[test]
fn drain_waits_for_pending_dispatch() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let drain_registry = registry.clone();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || done_tx.send(drain_registry.drain(short_budget())).unwrap());
    assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
    pending.end();
    assert_eq!(
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
}

#[test]
fn wait_pending_does_not_drain_active_scope() {
    let registry = Registry::new();
    let active_pending = registry.begin_dispatch().unwrap();
    let scope = registry
        .install(&active_pending, ScopeSpec::default())
        .unwrap();
    let pending = registry.begin_dispatch().unwrap();
    let wait_registry = registry.clone();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || {
        done_tx
            .send(wait_registry.wait_pending(short_budget()))
            .unwrap();
    });
    assert!(done_rx.recv_timeout(Duration::from_millis(20)).is_err());
    pending.end();
    assert_eq!(
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
    registry.begin_dispatch().unwrap().end();
    scope.end("test cleanup");
}

#[test]
fn drain_returns_when_blocking_resource_close_exceeds_budget() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry.install(&pending, ScopeSpec::default()).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scope
        .bind(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();

    assert_eq!(
        registry.drain(WaitBudget::for_duration(Duration::from_millis(20))),
        Err(RegistryError::DeadlineExceeded)
    );
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(registry.state(), State::Draining);

    let end_scope = scope.clone();
    let (ended_tx, ended_rx) = mpsc::channel();
    std::thread::spawn(move || {
        end_scope.end("canceled");
        ended_tx.send(()).unwrap();
    });
    release_tx.send(()).unwrap();
    ended_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    registry.drain(WaitBudget::unbounded()).unwrap();
}

#[test]
fn drain_waits_for_blocking_resource_close() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry.install(&pending, ScopeSpec::default()).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scope
        .bind(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();

    let drain_registry = registry.clone();
    let (drain_tx, drain_rx) = mpsc::channel();
    std::thread::spawn(move || drain_tx.send(drain_registry.drain(short_budget())).unwrap());
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let end_scope = scope.clone();
    std::thread::spawn(move || end_scope.end("canceled"));
    assert!(drain_rx.recv_timeout(Duration::from_millis(20)).is_err());
    release_tx.send(()).unwrap();
    assert_eq!(
        drain_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
}

#[test]
fn concurrent_drain_waits_for_blocking_resource_close() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry.install(&pending, ScopeSpec::default()).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scope
        .bind(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();

    let first_registry = registry.clone();
    let (first_tx, first_rx) = mpsc::channel();
    std::thread::spawn(move || first_tx.send(first_registry.drain(short_budget())).unwrap());
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();

    let end_scope = scope.clone();
    let (ended_tx, ended_rx) = mpsc::channel();
    std::thread::spawn(move || {
        end_scope.end("canceled");
        ended_tx.send(()).unwrap();
    });
    assert!(ended_rx.recv_timeout(Duration::from_millis(20)).is_err());

    let second_registry = registry.clone();
    let (second_tx, second_rx) = mpsc::channel();
    std::thread::spawn(move || {
        second_tx
            .send(second_registry.drain(short_budget()))
            .unwrap()
    });
    assert!(second_rx.recv_timeout(Duration::from_millis(20)).is_err());
    release_tx.send(()).unwrap();
    ended_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    assert_eq!(
        first_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
    assert_eq!(
        second_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
}

#[test]
fn concurrent_close_waits_for_blocking_resource_close() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry.install(&pending, ScopeSpec::default()).unwrap();
    let (started_tx, started_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    scope
        .bind(move || {
            started_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            Ok(())
        })
        .unwrap();

    let first_registry = registry.clone();
    let (first_tx, first_rx) = mpsc::channel();
    std::thread::spawn(move || first_tx.send(first_registry.close()).unwrap());
    started_rx.recv_timeout(Duration::from_secs(1)).unwrap();
    let second_registry = registry.clone();
    let (second_tx, second_rx) = mpsc::channel();
    std::thread::spawn(move || second_tx.send(second_registry.close()).unwrap());
    assert!(second_rx.recv_timeout(Duration::from_millis(20)).is_err());
    release_tx.send(()).unwrap();
    assert_eq!(
        first_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
    assert_eq!(
        second_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
}

#[test]
fn drain_rejects_late_bind() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry.install(&pending, ScopeSpec::default()).unwrap();
    let drain_registry = registry.clone();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || done_tx.send(drain_registry.drain(short_budget())).unwrap());
    wait_for_state(&registry, State::Draining);
    assert_eq!(scope.bind(|| Ok(())), Err(RegistryError::NotAccepting));
    scope.end("canceled");
    assert_eq!(
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
}

#[test]
fn drain_rejects_late_install() {
    let registry = Registry::new();
    let pending = registry.begin_dispatch().unwrap();
    let drain_registry = registry.clone();
    let (done_tx, done_rx) = mpsc::channel();
    std::thread::spawn(move || done_tx.send(drain_registry.drain(short_budget())).unwrap());
    wait_for_state(&registry, State::Draining);
    assert!(matches!(
        registry.install(&pending, ScopeSpec::default()),
        Err(RegistryError::NotAccepting)
    ));
    assert_eq!(
        done_rx.recv_timeout(Duration::from_secs(1)).unwrap(),
        Ok(())
    );
}

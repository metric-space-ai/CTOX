// ref: internal/runtime/executor/websocket_session_target_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::codex_websockets_connection::{
    CodexWebsocketActionFuture, CodexWebsocketConnection, CodexWebsocketFrame,
    CodexWebsocketFrameFuture,
};
use super::codex_websockets_errors::CodexWebsocketError;
use super::codex_websockets_executor::bind_connection_lifecycle;
use super::codex_websockets_session::CodexWebsocketSessionStore;
use crate::sdk::cliproxy::executor::{
    BoundResourceCloser, ExecutionLifecycle, LifecycleResult, ResourceCloseFn,
};

#[derive(Default)]
struct PhysicalClose {
    closed: AtomicBool,
    calls: AtomicUsize,
}

impl PhysicalClose {
    fn close(&self) {
        if !self.closed.swap(true, Ordering::AcqRel) {
            self.calls.fetch_add(1, Ordering::SeqCst);
        }
    }
}

struct Connection(Arc<PhysicalClose>);

impl CodexWebsocketConnection for Connection {
    fn send<'a>(&'a mut self, _: CodexWebsocketFrame) -> CodexWebsocketActionFuture<'a> {
        Box::pin(async { Ok(()) })
    }

    fn receive<'a>(&'a mut self) -> CodexWebsocketFrameFuture<'a> {
        Box::pin(async { Err(CodexWebsocketError::protocol("closed", true)) })
    }

    fn close<'a>(&'a mut self) -> CodexWebsocketActionFuture<'a> {
        let physical = Arc::clone(&self.0);
        Box::pin(async move {
            physical.close();
            Ok(())
        })
    }

    fn lifecycle_closer(&self) -> Option<ResourceCloseFn> {
        let physical = Arc::clone(&self.0);
        Some(Box::new(move || {
            physical.close();
            Ok(())
        }))
    }
}

#[derive(Default)]
struct Lifecycle {
    closer: Mutex<Option<BoundResourceCloser>>,
    binds: AtomicUsize,
    ends: AtomicUsize,
}

impl ExecutionLifecycle for Lifecycle {
    fn bind(&self, closer: BoundResourceCloser) -> LifecycleResult {
        self.binds.fetch_add(1, Ordering::SeqCst);
        *self.closer.lock().unwrap() = Some(closer);
        Ok(())
    }

    fn end(&self, _: &str) {
        self.ends.fetch_add(1, Ordering::SeqCst);
        if let Some(closer) = self.closer.lock().unwrap().take() {
            closer.close().unwrap();
        }
    }
}

#[tokio::test]
async fn session_target_and_lifecycle_belong_to_one_generation() {
    let store = CodexWebsocketSessionStore::default();
    let session = store.get_or_create("session-a").unwrap();
    let physical = Arc::new(PhysicalClose::default());
    let lifecycle = Arc::new(Lifecycle::default());
    {
        let mut state = session.execution.lock().await;
        state.auth_id = "auth-a".into();
        state.target_url = "wss://first.example/responses".into();
        state.generation = 1;
        state.connection = Some(Box::new(Connection(Arc::clone(&physical))));
        let lifecycle_trait: Arc<dyn ExecutionLifecycle> = lifecycle.clone();
        bind_connection_lifecycle(&mut state, Some(&lifecycle_trait)).unwrap();
        bind_connection_lifecycle(&mut state, Some(&lifecycle_trait)).unwrap();
        assert_eq!(state.lifecycle_generation, state.generation);
    }
    assert_eq!(lifecycle.binds.load(Ordering::SeqCst), 1);
    store.close("session-a").await;
    assert_eq!(lifecycle.ends.load(Ordering::SeqCst), 1);
    assert_eq!(physical.calls.load(Ordering::SeqCst), 1);
    assert!(store.is_empty());
}

#[tokio::test]
async fn stale_session_close_cannot_remove_replacement_session() {
    let store = CodexWebsocketSessionStore::default();
    let original = store.get_or_create("same-id").unwrap();
    let removed = store.remove("same-id").unwrap();
    assert!(Arc::ptr_eq(&original, &removed));
    let replacement = store.get_or_create("same-id").unwrap();
    assert!(!Arc::ptr_eq(&original, &replacement));

    let original_physical = Arc::new(PhysicalClose::default());
    original.execution.lock().await.connection =
        Some(Box::new(Connection(Arc::clone(&original_physical))));
    original.execution.lock().await.close("stale").await;
    assert_eq!(original_physical.calls.load(Ordering::SeqCst), 1);
    assert!(Arc::ptr_eq(
        &replacement,
        &store.get_or_create("same-id").unwrap()
    ));
}

#[tokio::test]
async fn close_all_is_instance_owned_and_physically_closes_each_target_once() {
    let first = CodexWebsocketSessionStore::default();
    let second = CodexWebsocketSessionStore::default();
    let close_a = Arc::new(PhysicalClose::default());
    let close_b = Arc::new(PhysicalClose::default());
    first
        .get_or_create("a")
        .unwrap()
        .execution
        .lock()
        .await
        .connection = Some(Box::new(Connection(Arc::clone(&close_a))));
    first
        .get_or_create("b")
        .unwrap()
        .execution
        .lock()
        .await
        .connection = Some(Box::new(Connection(Arc::clone(&close_b))));
    second.get_or_create("a").unwrap();

    first.close_all().await;
    first.close_all().await;
    assert_eq!(close_a.calls.load(Ordering::SeqCst), 1);
    assert_eq!(close_b.calls.load(Ordering::SeqCst), 1);
    assert!(first.is_empty());
    assert_eq!(second.len(), 1);
}

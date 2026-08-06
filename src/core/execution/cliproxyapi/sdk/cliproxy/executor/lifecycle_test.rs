// ref: sdk/cliproxy/executor/lifecycle_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::{
    bind_execution_resource, BindAndCloseError, BoundResourceCloser, ExecutionLifecycle,
    LifecycleError, LifecycleResult,
};

#[derive(Debug)]
struct TestError(&'static str);

impl fmt::Display for TestError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.0)
    }
}

impl Error for TestError {}

#[derive(Default)]
struct LifecycleRecorder {
    closer: Mutex<Option<BoundResourceCloser>>,
}

impl ExecutionLifecycle for LifecycleRecorder {
    fn bind(&self, closer: BoundResourceCloser) -> LifecycleResult {
        *self.closer.lock().unwrap() = Some(closer);
        Ok(())
    }

    fn end(&self, _reason: &str) {}
}

#[test]
fn bound_execution_resource_closes_once_across_cloned_handles() {
    let lifecycle = LifecycleRecorder::default();
    let calls = Arc::new(AtomicUsize::new(0));
    let close_calls = Arc::clone(&calls);
    bind_execution_resource(
        Some(&lifecycle),
        Some(Box::new(move || {
            close_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })),
    )
    .unwrap();

    let closer = lifecycle.closer.lock().unwrap().clone().unwrap();
    let cloned = closer.clone();
    assert!(closer.close().is_ok());
    assert!(cloned.close().is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

struct FailingLifecycle {
    error: LifecycleError,
}

impl ExecutionLifecycle for FailingLifecycle {
    fn bind(&self, _closer: BoundResourceCloser) -> LifecycleResult {
        Err(Arc::clone(&self.error))
    }

    fn end(&self, _reason: &str) {}
}

#[test]
fn bind_failure_closes_immediately_and_preserves_bind_error() {
    let bind_error: LifecycleError = Arc::new(TestError("selection ended"));
    let lifecycle = FailingLifecycle {
        error: Arc::clone(&bind_error),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let close_calls = Arc::clone(&calls);

    let error = bind_execution_resource(
        Some(&lifecycle),
        Some(Box::new(move || {
            close_calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })),
    )
    .unwrap_err();

    assert!(Arc::ptr_eq(&error, &bind_error));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn bind_and_close_failures_are_joined_without_double_close() {
    let lifecycle = FailingLifecycle {
        error: Arc::new(TestError("bind failed")),
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let close_calls = Arc::clone(&calls);

    let error = bind_execution_resource(
        Some(&lifecycle),
        Some(Box::new(move || {
            close_calls.fetch_add(1, Ordering::SeqCst);
            Err(Arc::new(TestError("close failed")))
        })),
    )
    .unwrap_err();
    let joined = error.downcast_ref::<BindAndCloseError>().unwrap();

    assert_eq!(joined.bind_error().to_string(), "bind failed");
    assert_eq!(joined.close_error().to_string(), "close failed");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn missing_lifecycle_or_resource_is_a_noop() {
    assert!(bind_execution_resource(None, None).is_ok());
    let lifecycle = LifecycleRecorder::default();
    assert!(bind_execution_resource(Some(&lifecycle), None).is_ok());
}

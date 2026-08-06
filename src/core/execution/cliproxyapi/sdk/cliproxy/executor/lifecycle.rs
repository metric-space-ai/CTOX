// ref: sdk/cliproxy/executor/lifecycle.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

pub type LifecycleError = Arc<dyn Error + Send + Sync + 'static>;
pub type LifecycleResult = Result<(), LifecycleError>;
pub type ResourceCloseFn = Box<dyn FnOnce() -> LifecycleResult + Send + 'static>;

/// Owns resources associated with one execution attempt.
pub trait ExecutionLifecycle: Send + Sync {
    fn bind(&self, closer: BoundResourceCloser) -> LifecycleResult;
    fn end(&self, reason: &str);
}

struct BoundResourceState {
    closer: Mutex<Option<ResourceCloseFn>>,
    result: OnceLock<LifecycleResult>,
}

/// A clonable handle that runs its resource closer at most once and replays the
/// same close result to every caller.
#[derive(Clone)]
pub struct BoundResourceCloser {
    state: Arc<BoundResourceState>,
}

impl BoundResourceCloser {
    fn new(closer: ResourceCloseFn) -> Self {
        Self {
            state: Arc::new(BoundResourceState {
                closer: Mutex::new(Some(closer)),
                result: OnceLock::new(),
            }),
        }
    }

    pub fn close(&self) -> LifecycleResult {
        self.state
            .result
            .get_or_init(|| {
                let closer = self
                    .state
                    .closer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .take();
                match closer {
                    Some(closer) => closer(),
                    None => Ok(()),
                }
            })
            .clone()
    }
}

impl fmt::Debug for BoundResourceCloser {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BoundResourceCloser")
            .field("closed", &self.state.result.get().is_some())
            .finish()
    }
}

/// Preserves both the lifecycle bind failure and a cleanup failure.
#[derive(Debug)]
pub struct BindAndCloseError {
    bind: LifecycleError,
    close: LifecycleError,
}

impl BindAndCloseError {
    pub fn bind_error(&self) -> &(dyn Error + Send + Sync + 'static) {
        self.bind.as_ref()
    }

    pub fn close_error(&self) -> &(dyn Error + Send + Sync + 'static) {
        self.close.as_ref()
    }
}

impl fmt::Display for BindAndCloseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "execution lifecycle bind failed: {}; resource close failed: {}",
            self.bind, self.close
        )
    }
}

impl Error for BindAndCloseError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        Some(self.bind.as_ref())
    }
}

/// Binds a resource closer to an execution lifecycle.
///
/// Missing lifecycle or resource is a no-op. If binding fails, the resource is
/// closed immediately. A simultaneous close error is joined rather than hiding
/// either failure.
pub fn bind_execution_resource(
    lifecycle: Option<&dyn ExecutionLifecycle>,
    closer: Option<ResourceCloseFn>,
) -> LifecycleResult {
    let (Some(lifecycle), Some(closer)) = (lifecycle, closer) else {
        return Ok(());
    };
    let closer = BoundResourceCloser::new(closer);
    match lifecycle.bind(closer.clone()) {
        Ok(()) => Ok(()),
        Err(bind) => match closer.close() {
            Ok(()) => Err(bind),
            Err(close) => Err(Arc::new(BindAndCloseError { bind, close })),
        },
    }
}

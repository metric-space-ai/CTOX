// ref: internal/pluginhost/client_guard.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: guards process-client calls and shutdown instead of in-process calls
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use tokio::sync::{mpsc, Notify};

use super::abi::{PluginCall, PluginClient, PluginClientError, PluginFuture, PluginStream};

struct GuardState {
    inner: Option<Arc<dyn PluginClient>>,
    calls: usize,
    closed: bool,
    shutdown_started: bool,
    shutdown_complete: bool,
}

struct GuardShared {
    state: Mutex<GuardState>,
    changed: Notify,
}

#[derive(Clone)]
pub struct GuardedPluginClient {
    shared: Arc<GuardShared>,
}

impl std::fmt::Debug for GuardedPluginClient {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        formatter
            .debug_struct("GuardedPluginClient")
            .field("active_calls", &state.calls)
            .field("closed", &state.closed)
            .finish_non_exhaustive()
    }
}

impl GuardedPluginClient {
    pub fn new(inner: Arc<dyn PluginClient>) -> Self {
        Self {
            shared: Arc::new(GuardShared {
                state: Mutex::new(GuardState {
                    inner: Some(inner),
                    calls: 0,
                    closed: false,
                    shutdown_started: false,
                    shutdown_complete: false,
                }),
                changed: Notify::new(),
            }),
        }
    }

    fn acquire(&self) -> Result<(Arc<dyn PluginClient>, ActiveCall), PluginClientError> {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.closed {
            return Err(PluginClientError::Closed);
        }
        let inner = state.inner.clone().ok_or(PluginClientError::Closed)?;
        state.calls += 1;
        Ok((
            inner,
            ActiveCall {
                shared: Arc::clone(&self.shared),
            },
        ))
    }

    pub fn is_closed(&self) -> bool {
        self.shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .closed
    }
}

struct ActiveCall {
    shared: Arc<GuardShared>,
}

impl Drop for ActiveCall {
    fn drop(&mut self) {
        let mut state = self
            .shared
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        state.calls = state.calls.saturating_sub(1);
        drop(state);
        self.shared.changed.notify_waiters();
    }
}

impl PluginClient for GuardedPluginClient {
    fn call<'a>(&'a self, call: PluginCall) -> PluginFuture<'a, crate::sdk::pluginabi::Envelope> {
        Box::pin(async move {
            let (inner, _active) = self.acquire()?;
            inner.call(call).await
        })
    }

    fn call_stream<'a>(&'a self, call: PluginCall) -> PluginFuture<'a, PluginStream> {
        Box::pin(async move {
            let (inner, active) = self.acquire()?;
            let mut source = inner.call_stream(call).await?;
            let (sender, receiver) = mpsc::channel(32);
            tokio::spawn(async move {
                let _active = active;
                while let Some(chunk) = source.chunks.recv().await {
                    if sender.send(chunk).await.is_err() {
                        break;
                    }
                }
            });
            Ok(PluginStream { chunks: receiver })
        })
    }

    fn shutdown<'a>(&'a self) -> PluginFuture<'a, ()> {
        Box::pin(async move {
            let (inner, leader) = {
                let mut state = self
                    .shared
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                state.closed = true;
                if state.shutdown_started {
                    (None, false)
                } else {
                    state.shutdown_started = true;
                    (state.inner.take(), true)
                }
            };
            if !leader {
                loop {
                    let notified = self.shared.changed.notified();
                    if self
                        .shared
                        .state
                        .lock()
                        .unwrap_or_else(std::sync::PoisonError::into_inner)
                        .shutdown_complete
                    {
                        return Ok(());
                    }
                    notified.await;
                }
            }
            loop {
                let notified = self.shared.changed.notified();
                let calls = self
                    .shared
                    .state
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .calls;
                if calls == 0 {
                    break;
                }
                notified.await;
            }
            let result = match inner {
                Some(inner) => inner.shutdown().await,
                None => Ok(()),
            };
            self.shared
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .shutdown_complete = true;
            self.shared.changed.notify_waiters();
            result
        })
    }
}

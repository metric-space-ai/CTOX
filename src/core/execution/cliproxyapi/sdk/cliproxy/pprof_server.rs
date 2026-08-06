// ref: sdk/cliproxy/pprof_server.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Ownership-safe lifecycle for the optional profiling listener.
//!
//! Rust cannot expose Go runtime profiles. The listener authority receives the
//! exact upstream route set and may map it to host-native diagnostics.

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::providers::LoadContext;

pub const DEFAULT_PPROF_ADDR: &str = "127.0.0.1:6060";
pub const PPROF_ROUTES: [&str; 11] = [
    "/debug/pprof/",
    "/debug/pprof/cmdline",
    "/debug/pprof/profile",
    "/debug/pprof/symbol",
    "/debug/pprof/trace",
    "/debug/pprof/allocs",
    "/debug/pprof/block",
    "/debug/pprof/goroutine",
    "/debug/pprof/heap",
    "/debug/pprof/mutex",
    "/debug/pprof/threadcreate",
];

pub type ListenerFuture = Pin<Box<dyn Future<Output = Result<(), PprofError>> + Send>>;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PprofError {
    Bind,
    Serve,
    Shutdown,
}

impl fmt::Display for PprofError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bind => "pprof listener bind failed",
            Self::Serve => "pprof listener serve failed",
            Self::Shutdown => "pprof listener shutdown failed",
        })
    }
}

impl std::error::Error for PprofError {}

pub trait PprofListener: Send + Sync {
    fn serve(self: Arc<Self>) -> ListenerFuture;
    fn shutdown(&self, timeout: Duration) -> ListenerFuture;
}

pub trait PprofListenerFactory: Send + Sync {
    fn bind(
        &self,
        addr: &str,
        routes: &'static [&'static str],
    ) -> Result<Arc<dyn PprofListener>, PprofError>;
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PprofConfig {
    pub enable: bool,
    pub addr: String,
}

impl Default for PprofConfig {
    fn default() -> Self {
        Self {
            enable: false,
            addr: DEFAULT_PPROF_ADDR.to_owned(),
        }
    }
}

struct State {
    server: Option<Arc<dyn PprofListener>>,
    addr: String,
    enabled: bool,
    owner: u64,
}

pub struct PprofServer {
    state: Arc<Mutex<State>>,
    factory: Arc<dyn PprofListenerFactory>,
    stop_timeout: Duration,
}

impl PprofServer {
    #[must_use]
    pub fn new(factory: Arc<dyn PprofListenerFactory>, stop_timeout: Duration) -> Self {
        Self {
            state: Arc::new(Mutex::new(State {
                server: None,
                addr: String::new(),
                enabled: false,
                owner: 0,
            })),
            factory,
            stop_timeout,
        }
    }

    pub async fn apply_context(&self, cfg: &PprofConfig) -> bool {
        self.apply_context_with_cancellation(cfg, None).await
    }

    pub async fn apply_context_with_cancellation(
        &self,
        cfg: &PprofConfig,
        cancellation: Option<&LoadContext>,
    ) -> bool {
        if cancellation.is_some_and(LoadContext::is_cancelled) {
            return false;
        }
        let addr = if cfg.addr.trim().is_empty() {
            DEFAULT_PPROF_ADDR.to_owned()
        } else {
            cfg.addr.trim().to_owned()
        };
        let (current, current_addr, owner) = {
            let mut state = self.state.lock().unwrap();
            state.owner = state.owner.wrapping_add(1);
            let owner = state.owner;
            let current = state.server.clone();
            let current_addr = state.addr.clone();
            state.addr.clone_from(&addr);
            state.enabled = cfg.enable;
            if !cfg.enable || current.as_ref().is_none_or(|_| current_addr != addr) {
                state.server = None;
            }
            (current, current_addr, owner)
        };
        if !cfg.enable {
            return self.stop_server(current).await.is_ok();
        }
        if current.is_some() && current_addr == addr {
            return true;
        }
        if self.stop_server(current).await.is_err() {
            return false;
        }
        if cancellation.is_some_and(LoadContext::is_cancelled) {
            return false;
        }
        let Ok(server) = self.factory.bind(&addr, &PPROF_ROUTES) else {
            return false;
        };
        {
            let mut state = self.state.lock().unwrap();
            if !state.enabled
                || state.addr != addr
                || state.owner != owner
                || state.server.is_some()
            {
                return false;
            }
            state.server = Some(server.clone());
        }
        let state = self.state.clone();
        let monitored_server = server.clone();
        tokio::spawn(async move {
            if monitored_server.clone().serve().await.is_err() {
                clear_failed_state(&state, &monitored_server);
            }
        });
        if cancellation.is_some_and(LoadContext::is_cancelled) {
            let _ = self.stop_owned_server(server, owner).await;
            return false;
        }
        true
    }

    pub async fn shutdown(&self) -> Result<(), PprofError> {
        let current = {
            let mut state = self.state.lock().unwrap();
            state.owner = state.owner.wrapping_add(1);
            state.enabled = false;
            state.server.take()
        };
        self.stop_server(current).await
    }

    async fn stop_server(&self, server: Option<Arc<dyn PprofListener>>) -> Result<(), PprofError> {
        match server {
            Some(server) => server.shutdown(self.stop_timeout).await,
            None => Ok(()),
        }
    }

    pub(crate) async fn stop_owned_server(
        &self,
        server: Arc<dyn PprofListener>,
        owner: u64,
    ) -> Result<(), PprofError> {
        let should_stop = {
            let mut state = self.state.lock().unwrap();
            if state.owner == owner
                && state
                    .server
                    .as_ref()
                    .is_some_and(|current| Arc::ptr_eq(current, &server))
            {
                state.server = None;
                true
            } else {
                false
            }
        };
        if should_stop {
            server.shutdown(self.stop_timeout).await
        } else {
            Ok(())
        }
    }

    #[cfg(test)]
    pub(crate) fn clear_failed_server(&self, server: &Arc<dyn PprofListener>) {
        clear_failed_state(&self.state, server);
    }

    #[cfg(test)]
    pub(crate) fn seed(
        &self,
        server: Arc<dyn PprofListener>,
        addr: &str,
        enabled: bool,
        owner: u64,
    ) {
        *self.state.lock().unwrap() = State {
            server: Some(server),
            addr: addr.to_owned(),
            enabled,
            owner,
        };
    }

    #[cfg(test)]
    pub(crate) fn snapshot(&self) -> (Option<Arc<dyn PprofListener>>, u64) {
        let state = self.state.lock().unwrap();
        (state.server.clone(), state.owner)
    }
}

fn clear_failed_state(state: &Mutex<State>, server: &Arc<dyn PprofListener>) {
    let mut state = state.lock().unwrap();
    if state
        .server
        .as_ref()
        .is_some_and(|current| Arc::ptr_eq(current, server))
    {
        state.server = None;
    }
}

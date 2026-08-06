// ref: internal/wsrelay/manager.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use futures_util::future::join_all;
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::mpsc;
use tokio_tungstenite::WebSocketStream;

use super::http::{RelayAuthority, RelayHandshake};
use super::session::{
    writer_loop, RelayCancellation, RelayClock, RelayError, RelayTransport, Session,
    WebSocketTransport,
};
use super::Message;

#[derive(Clone, Debug)]
pub struct RelayLimits {
    pub outbound_capacity: usize,
    pub response_capacity: usize,
    pub max_inbound_message_len: usize,
    pub read_timeout: Duration,
    pub write_timeout: Duration,
    pub heartbeat_interval: Duration,
    pub shutdown_timeout: Duration,
}

impl Default for RelayLimits {
    fn default() -> Self {
        Self {
            outbound_capacity: 32,
            response_capacity: 8,
            max_inbound_message_len: 64 << 20,
            read_timeout: Duration::from_secs(60),
            write_timeout: Duration::from_secs(10),
            heartbeat_interval: Duration::from_secs(30),
            shutdown_timeout: Duration::from_secs(10),
        }
    }
}

impl RelayLimits {
    fn validate(&self) -> Result<(), RelayError> {
        if self.outbound_capacity == 0 || self.response_capacity == 0 {
            return Err(RelayError::InvalidRequest(
                "relay queue capacities must be positive".into(),
            ));
        }
        if self.max_inbound_message_len == 0 {
            return Err(RelayError::InvalidRequest(
                "maximum inbound message length must be positive".into(),
            ));
        }
        for (name, duration) in [
            ("read timeout", self.read_timeout),
            ("write timeout", self.write_timeout),
            ("heartbeat interval", self.heartbeat_interval),
            ("shutdown timeout", self.shutdown_timeout),
        ] {
            if duration.is_zero() {
                return Err(RelayError::InvalidRequest(format!(
                    "{name} must be positive"
                )));
            }
        }
        Ok(())
    }
}

pub trait RelayEventSink: Send + Sync {
    fn connected(&self, _provider: &str) {}
    fn disconnected(&self, _provider: &str, _cause: &RelayError) {}
    fn unknown_terminal_message(&self, _provider: &str, _message_id: &str) {}
}

struct NoopEventSink;

impl RelayEventSink for NoopEventSink {}

pub struct SystemRelayClock;

impl RelayClock for SystemRelayClock {
    fn utc_now(&self) -> DateTime<Utc> {
        std::time::SystemTime::now().into()
    }

    fn monotonic_now(&self) -> Instant {
        Instant::now()
    }
}

pub struct ManagerOptions {
    pub path: String,
    pub authority: Arc<dyn RelayAuthority>,
    pub events: Option<Arc<dyn RelayEventSink>>,
    pub clock: Option<Arc<dyn RelayClock>>,
    pub limits: RelayLimits,
}

#[derive(Clone)]
pub struct Manager {
    pub(crate) inner: Arc<ManagerInner>,
}

pub(crate) struct ManagerInner {
    path: String,
    authority: Arc<dyn RelayAuthority>,
    pub(crate) events: Arc<dyn RelayEventSink>,
    pub(crate) clock: Arc<dyn RelayClock>,
    pub(crate) limits: RelayLimits,
    sessions: Mutex<HashMap<String, Arc<Session>>>,
    stopped: AtomicBool,
}

impl Manager {
    pub fn new(options: ManagerOptions) -> Result<Self, RelayError> {
        options.limits.validate()?;
        let path = normalize_path(&options.path);
        Ok(Self {
            inner: Arc::new(ManagerInner {
                path,
                authority: options.authority,
                events: options.events.unwrap_or_else(|| Arc::new(NoopEventSink)),
                clock: options.clock.unwrap_or_else(|| Arc::new(SystemRelayClock)),
                limits: options.limits,
                sessions: Mutex::new(HashMap::new()),
                stopped: AtomicBool::new(false),
            }),
        })
    }

    pub fn path(&self) -> &str {
        &self.inner.path
    }

    pub fn is_connected(&self, provider: &str) -> bool {
        self.inner
            .sessions
            .lock()
            .expect("wsrelay sessions poisoned")
            .contains_key(&normalize_provider(provider))
    }

    pub fn connected_providers(&self) -> Vec<String> {
        let mut providers: Vec<_> = self
            .inner
            .sessions
            .lock()
            .expect("wsrelay sessions poisoned")
            .keys()
            .cloned()
            .collect();
        providers.sort();
        providers
    }

    pub async fn accept_transport(
        &self,
        handshake: RelayHandshake,
        transport: Box<dyn RelayTransport>,
    ) -> Result<String, RelayError> {
        if self.inner.stopped.load(Ordering::Acquire) {
            return Err(RelayError::Closed);
        }
        if !handshake.method.eq_ignore_ascii_case("GET") {
            return Err(RelayError::InvalidRequest(
                "websocket upgrade method must be GET".into(),
            ));
        }
        if handshake.path != self.inner.path {
            return Err(RelayError::InvalidRequest(
                "websocket upgrade path does not match relay path".into(),
            ));
        }
        let provider = normalize_provider(&self.inner.authority.authorize(&handshake)?);
        if provider.is_empty() {
            return Err(RelayError::Unauthorized(
                "authority returned an empty provider".into(),
            ));
        }
        let (sender, receiver) = transport.split();
        let (outbound_tx, outbound_rx) = mpsc::channel(self.inner.limits.outbound_capacity);
        let session = Arc::new(Session::new(
            provider.clone(),
            format!("aistudio-{}", uuid::Uuid::new_v4().simple()),
            Arc::downgrade(&self.inner),
            outbound_tx,
            self.inner.limits.response_capacity,
        ));
        let replaced = {
            let mut sessions = self
                .inner
                .sessions
                .lock()
                .expect("wsrelay sessions poisoned");
            if self.inner.stopped.load(Ordering::Acquire) {
                return Err(RelayError::Closed);
            }
            sessions.insert(provider.clone(), Arc::clone(&session))
        };
        if let Some(replaced) = replaced {
            replaced.cancel(RelayError::Replaced);
        }

        let (error_tx, mut error_rx) = mpsc::channel(1);
        let writer_cancellation = session.cancellation();
        let write_timeout = self.inner.limits.write_timeout;
        self.inner.events.connected(&provider);
        tokio::spawn(writer_loop(
            sender,
            outbound_rx,
            writer_cancellation,
            write_timeout,
            error_tx,
        ));
        let run_session = Arc::clone(&session);
        let limits = self.inner.limits.clone();
        let clock = Arc::clone(&self.inner.clock);
        tokio::spawn(async move {
            run_session
                .run(
                    receiver,
                    limits.read_timeout,
                    limits.heartbeat_interval,
                    &mut error_rx,
                    clock,
                )
                .await;
        });
        Ok(provider)
    }

    pub async fn accept_websocket<S>(
        &self,
        handshake: RelayHandshake,
        stream: WebSocketStream<S>,
    ) -> Result<String, RelayError>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        let transport = WebSocketTransport::new(stream, self.inner.limits.max_inbound_message_len);
        self.accept_transport(handshake, Box::new(transport)).await
    }

    pub async fn stop(&self, cancellation: RelayCancellation) -> Result<(), RelayError> {
        self.inner.stopped.store(true, Ordering::Release);
        let sessions: Vec<_> = self
            .inner
            .sessions
            .lock()
            .expect("wsrelay sessions poisoned")
            .drain()
            .map(|(_, session)| session)
            .collect();
        for session in &sessions {
            session.cancel(RelayError::ManagerStopped);
        }
        let waits = sessions.iter().map(|session| session.wait_finished());
        tokio::select! {
            () = cancellation.cancelled() => Err(RelayError::Cancelled),
            result = tokio::time::timeout(self.inner.limits.shutdown_timeout, join_all(waits)) => {
                result.map(|_| ()).map_err(|_| RelayError::TimedOut("shutdown"))
            }
        }
    }

    pub(crate) fn session(&self, provider: &str) -> Option<Arc<Session>> {
        self.inner
            .sessions
            .lock()
            .expect("wsrelay sessions poisoned")
            .get(&normalize_provider(provider))
            .cloned()
    }

    pub async fn send(
        &self,
        cancellation: RelayCancellation,
        provider: &str,
        message: Message,
    ) -> Result<mpsc::Receiver<Message>, RelayError> {
        let session = self
            .session(provider)
            .ok_or_else(|| RelayError::NotConnected(provider.to_owned()))?;
        session.request(cancellation, message).await
    }
}

impl ManagerInner {
    pub(crate) fn handle_session_closed(&self, session: &Session, cause: RelayError) {
        let mut sessions = self.sessions.lock().expect("wsrelay sessions poisoned");
        let is_current = sessions
            .get(&session.provider)
            .is_some_and(|current| std::ptr::eq(current.as_ref(), session));
        if is_current {
            sessions.remove(&session.provider);
        }
        drop(sessions);
        self.events.disconnected(&session.provider, &cause);
    }
}

fn normalize_path(path: &str) -> String {
    let path = path.trim();
    if path.is_empty() {
        "/v1/ws".into()
    } else if path.starts_with('/') {
        path.into()
    } else {
        format!("/{path}")
    }
}

fn normalize_provider(provider: &str) -> String {
    provider.trim().to_lowercase()
}

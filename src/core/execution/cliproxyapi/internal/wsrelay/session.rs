// ref: internal/wsrelay/session.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

use chrono::{DateTime, Utc};
use futures_util::stream::{SplitSink, SplitStream};
use futures_util::{SinkExt, StreamExt};
use serde_json::{Map, Value};
use tokio::io::{AsyncRead, AsyncWrite};
use tokio::sync::{mpsc, Notify};
use tokio_tungstenite::tungstenite::Message as WebSocketMessage;
use tokio_tungstenite::WebSocketStream;

use super::manager::ManagerInner;
use super::message::{Message, MESSAGE_TYPE_ERROR, MESSAGE_TYPE_PING, MESSAGE_TYPE_PONG};

pub type RelayFuture<'a, T> = Pin<Box<dyn Future<Output = T> + Send + 'a>>;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RelayError {
    Closed,
    Replaced,
    ManagerStopped,
    Cancelled,
    InvalidRequest(String),
    Unauthorized(String),
    NotConnected(String),
    DuplicateRequest(String),
    Backpressure(String),
    Transport(String),
    Protocol(String),
    TimedOut(&'static str),
    Upstream { message: String, status: u16 },
}

impl fmt::Display for RelayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Closed => formatter.write_str("wsrelay: websocket session closed"),
            Self::Replaced => formatter.write_str("wsrelay: replaced by new connection"),
            Self::ManagerStopped => formatter.write_str("wsrelay: manager stopped"),
            Self::Cancelled => formatter.write_str("wsrelay: operation cancelled"),
            Self::InvalidRequest(message) => {
                write!(formatter, "wsrelay: invalid request: {message}")
            }
            Self::Unauthorized(message) => write!(formatter, "wsrelay: unauthorized: {message}"),
            Self::NotConnected(provider) => {
                write!(formatter, "wsrelay: provider {provider} not connected")
            }
            Self::DuplicateRequest(id) => write!(formatter, "wsrelay: duplicate message id {id}"),
            Self::Backpressure(scope) => write!(formatter, "wsrelay: {scope} queue is full"),
            Self::Transport(message) => write!(formatter, "wsrelay: transport: {message}"),
            Self::Protocol(message) => write!(formatter, "wsrelay: protocol: {message}"),
            Self::TimedOut(scope) => write!(formatter, "wsrelay: {scope} timed out"),
            Self::Upstream { message, status } => write!(formatter, "{message} (status={status})"),
        }
    }
}

impl std::error::Error for RelayError {}

#[derive(Clone, Default)]
pub struct RelayCancellation {
    inner: Arc<CancellationInner>,
}

#[derive(Default)]
struct CancellationInner {
    cancelled: AtomicBool,
    notify: Notify,
}

impl RelayCancellation {
    pub fn cancel(&self) {
        if !self.inner.cancelled.swap(true, Ordering::AcqRel) {
            self.inner.notify.notify_waiters();
        }
    }

    pub fn is_cancelled(&self) -> bool {
        self.inner.cancelled.load(Ordering::Acquire)
    }

    pub async fn cancelled(&self) {
        loop {
            let notified = self.inner.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

pub trait RelayClock: Send + Sync {
    fn utc_now(&self) -> DateTime<Utc>;
    fn monotonic_now(&self) -> Instant;
}

pub trait RelayReceiver: Send {
    fn receive<'a>(&'a mut self) -> RelayFuture<'a, Result<Option<Message>, RelayError>>;
}

pub trait RelaySender: Send {
    fn send<'a>(&'a mut self, message: Message) -> RelayFuture<'a, Result<(), RelayError>>;
    fn ping<'a>(&'a mut self) -> RelayFuture<'a, Result<(), RelayError>>;
    fn close<'a>(&'a mut self) -> RelayFuture<'a, Result<(), RelayError>>;
}

/// Typed transport boundary: the host owns HTTP/WebSocket upgrade and hands
/// the resulting connection to the relay. This keeps authority and listener
/// policy outside the protocol engine.
pub trait RelayTransport: Send {
    fn split(self: Box<Self>) -> (Box<dyn RelaySender>, Box<dyn RelayReceiver>);
}

pub struct WebSocketTransport<S> {
    stream: WebSocketStream<S>,
    max_inbound_message_len: usize,
}

impl<S> WebSocketTransport<S> {
    pub fn new(stream: WebSocketStream<S>, max_inbound_message_len: usize) -> Self {
        Self {
            stream,
            max_inbound_message_len,
        }
    }
}

impl<S> RelayTransport for WebSocketTransport<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    fn split(self: Box<Self>) -> (Box<dyn RelaySender>, Box<dyn RelayReceiver>) {
        let (sink, stream) = self.stream.split();
        (
            Box::new(WebSocketSender { sink }),
            Box::new(WebSocketReceiver {
                stream,
                max_inbound_message_len: self.max_inbound_message_len,
            }),
        )
    }
}

struct WebSocketSender<S> {
    sink: SplitSink<WebSocketStream<S>, WebSocketMessage>,
}

impl<S> RelaySender for WebSocketSender<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    fn send<'a>(&'a mut self, message: Message) -> RelayFuture<'a, Result<(), RelayError>> {
        Box::pin(async move {
            let json = serde_json::to_string(&message)
                .map_err(|error| RelayError::Protocol(error.to_string()))?;
            self.sink
                .send(WebSocketMessage::Text(json.into()))
                .await
                .map_err(|error| RelayError::Transport(error.to_string()))
        })
    }

    fn ping<'a>(&'a mut self) -> RelayFuture<'a, Result<(), RelayError>> {
        Box::pin(async move {
            self.sink
                .send(WebSocketMessage::Ping(b"ping".as_slice().into()))
                .await
                .map_err(|error| RelayError::Transport(error.to_string()))
        })
    }

    fn close<'a>(&'a mut self) -> RelayFuture<'a, Result<(), RelayError>> {
        Box::pin(async move {
            self.sink
                .close()
                .await
                .map_err(|error| RelayError::Transport(error.to_string()))
        })
    }
}

struct WebSocketReceiver<S> {
    stream: SplitStream<WebSocketStream<S>>,
    max_inbound_message_len: usize,
}

impl<S> RelayReceiver for WebSocketReceiver<S>
where
    S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    fn receive<'a>(&'a mut self) -> RelayFuture<'a, Result<Option<Message>, RelayError>> {
        Box::pin(async move {
            loop {
                let frame = match self.stream.next().await {
                    Some(Ok(frame)) => frame,
                    Some(Err(error)) => return Err(RelayError::Transport(error.to_string())),
                    None => return Ok(None),
                };
                let bytes = match frame {
                    WebSocketMessage::Text(text) => text.as_bytes().to_vec(),
                    WebSocketMessage::Binary(bytes) => bytes.to_vec(),
                    WebSocketMessage::Close(_) => return Ok(None),
                    WebSocketMessage::Ping(_) | WebSocketMessage::Pong(_) => continue,
                    WebSocketMessage::Frame(_) => continue,
                };
                if bytes.len() > self.max_inbound_message_len {
                    return Err(RelayError::Protocol("inbound message exceeds limit".into()));
                }
                return serde_json::from_slice(&bytes)
                    .map(Some)
                    .map_err(|error| RelayError::Protocol(error.to_string()));
            }
        })
    }
}

pub(crate) enum Outbound {
    Message(Message),
    Ping,
}

pub(crate) struct Session {
    pub(crate) provider: String,
    _id: String,
    manager: Weak<ManagerInner>,
    outbound: mpsc::Sender<Outbound>,
    pending: Mutex<HashMap<String, mpsc::Sender<Message>>>,
    cancellation: RelayCancellation,
    requested_close: Mutex<Option<RelayError>>,
    finished: AtomicBool,
    finished_notify: Notify,
    response_capacity: usize,
}

impl Session {
    pub(crate) fn new(
        provider: String,
        id: String,
        manager: Weak<ManagerInner>,
        outbound: mpsc::Sender<Outbound>,
        response_capacity: usize,
    ) -> Self {
        Self {
            provider,
            _id: id,
            manager,
            outbound,
            pending: Mutex::new(HashMap::new()),
            cancellation: RelayCancellation::default(),
            requested_close: Mutex::new(None),
            finished: AtomicBool::new(false),
            finished_notify: Notify::new(),
            response_capacity,
        }
    }

    pub(crate) async fn request(
        self: &Arc<Self>,
        cancellation: RelayCancellation,
        message: Message,
    ) -> Result<mpsc::Receiver<Message>, RelayError> {
        if message.id.is_empty() {
            return Err(RelayError::InvalidRequest("message id is required".into()));
        }
        if self.cancellation.is_cancelled() {
            return Err(RelayError::Closed);
        }
        let (sender, receiver) = mpsc::channel(self.response_capacity);
        let watch_sender = sender.clone();
        {
            let mut pending = self.pending.lock().expect("wsrelay pending poisoned");
            match pending.entry(message.id.clone()) {
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(sender);
                }
                std::collections::hash_map::Entry::Occupied(_) => {
                    return Err(RelayError::DuplicateRequest(message.id));
                }
            }
        }
        let id = message.id.clone();
        let send_result = tokio::select! {
            () = cancellation.cancelled() => Err(RelayError::Cancelled),
            () = self.cancellation.cancelled() => Err(RelayError::Closed),
            result = self.outbound.send(Outbound::Message(message)) => {
                result.map_err(|_| RelayError::Closed)
            }
        };
        if let Err(error) = send_result {
            self.remove_pending(&id);
            return Err(error);
        }
        let weak = Arc::downgrade(self);
        let session_cancellation = self.cancellation();
        tokio::spawn(async move {
            let remove_pending = tokio::select! {
                () = cancellation.cancelled() => true,
                () = watch_sender.closed() => true,
                () = session_cancellation.cancelled() => false,
            };
            if remove_pending {
                if let Some(session) = weak.upgrade() {
                    session.remove_pending(&id);
                }
            }
        });
        Ok(receiver)
    }

    pub(crate) fn cancel(&self, cause: RelayError) {
        let mut requested = self
            .requested_close
            .lock()
            .expect("wsrelay close cause poisoned");
        if requested.is_none() {
            *requested = Some(cause);
        }
        drop(requested);
        self.cancellation.cancel();
    }

    pub(crate) fn cancellation(&self) -> RelayCancellation {
        self.cancellation.clone()
    }

    pub(crate) async fn wait_finished(&self) {
        loop {
            let notified = self.finished_notify.notified();
            if self.finished.load(Ordering::Acquire) {
                return;
            }
            notified.await;
        }
    }

    pub(crate) async fn run(
        self: Arc<Self>,
        mut receiver: Box<dyn RelayReceiver>,
        read_timeout: Duration,
        heartbeat_interval: Duration,
        transport_errors: &mut mpsc::Receiver<RelayError>,
        clock: Arc<dyn RelayClock>,
    ) {
        let mut heartbeat = tokio::time::interval(heartbeat_interval);
        heartbeat.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
        heartbeat.tick().await;
        let mut last_inbound = clock.monotonic_now();
        let cause = loop {
            tokio::select! {
                () = self.cancellation.cancelled() => {
                    break self
                        .requested_close
                        .lock()
                        .expect("wsrelay close cause poisoned")
                        .clone()
                        .unwrap_or(RelayError::Closed);
                }
                error = transport_errors.recv() => {
                    break error.unwrap_or(RelayError::Closed);
                }
                _ = heartbeat.tick() => {
                    if clock.monotonic_now().saturating_duration_since(last_inbound) >= read_timeout {
                        break RelayError::TimedOut("read");
                    }
                    if self.outbound.try_send(Outbound::Ping).is_err() {
                        break RelayError::Backpressure("outbound heartbeat".into());
                    }
                }
                inbound = receiver.receive() => {
                    match inbound {
                        Ok(Some(message)) => {
                            last_inbound = clock.monotonic_now();
                            if let Err(error) = self.dispatch(message) {
                                break error;
                            }
                        }
                        Ok(None) => break RelayError::Closed,
                        Err(error) => break error,
                    }
                }
            }
        };
        self.finish(cause);
    }

    fn dispatch(&self, message: Message) -> Result<(), RelayError> {
        if message.kind == MESSAGE_TYPE_PING {
            self.outbound
                .try_send(Outbound::Message(Message::new(
                    message.id,
                    MESSAGE_TYPE_PONG,
                )))
                .map_err(|_| RelayError::Backpressure("outbound".into()))?;
            return Ok(());
        }
        let terminal = message.is_terminal();
        let sender = {
            let mut pending = self.pending.lock().expect("wsrelay pending poisoned");
            if terminal {
                pending.remove(&message.id)
            } else {
                pending.get(&message.id).cloned()
            }
        };
        if let Some(sender) = sender {
            sender
                .try_send(message)
                .map_err(|_| RelayError::Backpressure("response".into()))?;
        } else if terminal {
            if let Some(manager) = self.manager.upgrade() {
                manager
                    .events
                    .unknown_terminal_message(&self.provider, &message.id);
            }
        }
        Ok(())
    }

    fn remove_pending(&self, id: &str) {
        self.pending
            .lock()
            .expect("wsrelay pending poisoned")
            .remove(id);
    }

    fn finish(&self, cause: RelayError) {
        if self.finished.swap(true, Ordering::AcqRel) {
            return;
        }
        self.cancellation.cancel();
        let pending = std::mem::take(&mut *self.pending.lock().expect("wsrelay pending poisoned"));
        for (id, sender) in pending {
            let mut payload = Map::new();
            payload.insert("error".into(), Value::String(cause.to_string()));
            let _ = sender.try_send(Message::with_payload(id, MESSAGE_TYPE_ERROR, payload));
        }
        if let Some(manager) = self.manager.upgrade() {
            manager.handle_session_closed(self, cause);
        }
        self.finished_notify.notify_waiters();
    }
}

pub(crate) async fn writer_loop(
    mut sender: Box<dyn RelaySender>,
    mut outbound: mpsc::Receiver<Outbound>,
    cancellation: RelayCancellation,
    write_timeout: Duration,
    errors: mpsc::Sender<RelayError>,
) {
    loop {
        let outbound = tokio::select! {
            () = cancellation.cancelled() => break,
            outbound = outbound.recv() => match outbound {
                Some(outbound) => outbound,
                None => break,
            }
        };
        let operation = match outbound {
            Outbound::Message(message) => sender.send(message),
            Outbound::Ping => sender.ping(),
        };
        match tokio::time::timeout(write_timeout, operation).await {
            Ok(Ok(())) => {}
            Ok(Err(error)) => {
                let _ = errors.try_send(error);
                break;
            }
            Err(_) => {
                let _ = errors.try_send(RelayError::TimedOut("write"));
                break;
            }
        }
    }
    let _ = tokio::time::timeout(write_timeout, sender.close()).await;
}

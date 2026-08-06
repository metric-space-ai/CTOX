// ref: internal/client/codex/live/media.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;
use std::future::Future;
use std::net::IpAddr;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use super::{LiveError, LiveErrorKind};

pub const OPUS_MIME_TYPE: &str = "audio/opus";
pub const OPUS_CLOCK_RATE: u32 = 48_000;
pub const OPUS_CHANNELS: u16 = 2;
pub const DATA_CHANNEL_LABEL: &str = "oai-events";
pub const MAX_PENDING_DATA_CHANNEL_MESSAGES: usize = 64;
pub const DATA_CHANNEL_BUFFER_LOW_THRESHOLD: u64 = 256 * 1024;
pub const MAX_DATA_CHANNEL_BUFFERED_AMOUNT: u64 = 1024 * 1024;

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct IceServer {
    pub urls: Vec<String>,
    pub username: String,
    pub credential: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LiveMediaRelayConfig {
    pub enabled: bool,
    pub max_sessions: usize,
    pub disable_private_remote_ips: bool,
    pub public_ip: Option<IpAddr>,
    pub udp_port_min: Option<u16>,
    pub udp_port_max: Option<u16>,
    pub ice_servers: Vec<IceServer>,
}

impl Default for LiveMediaRelayConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            max_sessions: 32,
            disable_private_remote_ips: false,
            public_ip: None,
            udp_port_min: None,
            udp_port_max: None,
            ice_servers: Vec::new(),
        }
    }
}

impl LiveMediaRelayConfig {
    pub fn validate(&self) -> Result<(), LiveError> {
        if self.max_sessions == 0 {
            return Err(LiveError::new(
                LiveErrorKind::Unavailable,
                "Codex live media max sessions must be positive",
            ));
        }
        match (self.udp_port_min, self.udp_port_max) {
            (Some(minimum), Some(maximum)) if minimum == 0 || maximum == 0 || minimum > maximum => {
                Err(LiveError::new(
                    LiveErrorKind::Unavailable,
                    "Codex live media UDP port range is invalid",
                ))
            }
            (Some(_), None) | (None, Some(_)) => Err(LiveError::new(
                LiveErrorKind::Unavailable,
                "Codex live media UDP port range must specify both bounds",
            )),
            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct MediaRoute {
    pub proxy_url: String,
    pub credential: String,
    pub auth_index: String,
}

impl MediaRoute {
    #[must_use]
    pub fn proxy_scheme(&self) -> String {
        super::proxy_scheme(&self.proxy_url)
    }

    /// Safe diagnostic rendering deliberately omits proxy URL credentials.
    #[must_use]
    pub fn diagnostic(&self) -> String {
        format!(
            "credential={} auth_index={} proxy={}",
            self.credential,
            self.auth_index,
            self.proxy_scheme()
        )
    }
}

pub type MediaFuture<'a, T> = Pin<Box<dyn Future<Output = Result<T, LiveError>> + Send + 'a>>;

/// Host-injected WebRTC implementation. The Rust port owns the exact lifecycle
/// contract while the embedding host may provide Pion-equivalent native media.
pub trait MediaRelay: Send + Sync {
    fn new_session<'a>(
        &'a self,
        client_offer: &'a str,
        route: &'a MediaRoute,
    ) -> MediaFuture<'a, (Arc<dyn MediaRelaySession>, String)>;
}

pub trait MediaRelaySession: Send + Sync {
    fn accept_upstream_answer<'a>(&'a self, answer: &'a str) -> MediaFuture<'a, String>;
    fn set_call_id(&self, call_id: &str);
    fn set_close_handler(&self, handler: Arc<dyn Fn(&str) + Send + Sync>);
    fn close(&self, reason: &str);
}

#[derive(Debug)]
pub struct MediaSessionLimiter {
    limit: AtomicUsize,
    active: AtomicUsize,
}

impl MediaSessionLimiter {
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            limit: AtomicUsize::new(limit),
            active: AtomicUsize::new(0),
        }
    }

    pub fn set_limit(&self, limit: usize) {
        self.limit.store(limit, Ordering::Release);
    }

    pub fn acquire(self: &Arc<Self>) -> Option<MediaSessionPermit> {
        loop {
            let active = self.active.load(Ordering::Acquire);
            let limit = self.limit.load(Ordering::Acquire);
            if limit == 0 || active >= limit {
                return None;
            }
            if self
                .active
                .compare_exchange(active, active + 1, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(MediaSessionPermit {
                    limiter: Arc::clone(self),
                    released: false,
                });
            }
        }
    }

    #[must_use]
    pub fn active(&self) -> usize {
        self.active.load(Ordering::Acquire)
    }
}

pub struct MediaSessionPermit {
    limiter: Arc<MediaSessionLimiter>,
    released: bool,
}

impl MediaSessionPermit {
    pub fn release(mut self) {
        self.release_once();
    }

    fn release_once(&mut self) {
        if !self.released {
            self.released = true;
            self.limiter.active.fetch_sub(1, Ordering::AcqRel);
        }
    }
}

impl Drop for MediaSessionPermit {
    fn drop(&mut self) {
        self.release_once();
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DataChannelMessage {
    pub binary: bool,
    pub data: Vec<u8>,
}

/// A bounded, transport-neutral version of the upstream data-channel pipe.
/// It preserves ordering, attaches late destinations, and rejects overflow.
pub struct DataChannelPipe {
    pending: Mutex<VecDeque<DataChannelMessage>>,
    destination: Mutex<Option<Arc<dyn DataChannelSink>>>,
    closed: AtomicBool,
}

pub trait DataChannelSink: Send + Sync {
    fn buffered_amount(&self) -> u64;
    fn send(&self, message: &DataChannelMessage) -> Result<(), LiveError>;
    fn close(&self);
}

impl DataChannelPipe {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pending: Mutex::new(VecDeque::new()),
            destination: Mutex::new(None),
            closed: AtomicBool::new(false),
        }
    }

    pub fn set_destination(&self, destination: Arc<dyn DataChannelSink>) -> Result<(), LiveError> {
        if self.closed.load(Ordering::Acquire) {
            destination.close();
            return Err(LiveError::new(
                LiveErrorKind::Unavailable,
                "Codex live data channel is closed",
            ));
        }
        *self
            .destination
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = Some(destination);
        self.flush()
    }

    pub fn push(&self, message: DataChannelMessage) -> Result<(), LiveError> {
        if self.closed.load(Ordering::Acquire) {
            return Err(LiveError::new(
                LiveErrorKind::Unavailable,
                "Codex live data channel is closed",
            ));
        }
        {
            let mut pending = self
                .pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if pending.len() >= MAX_PENDING_DATA_CHANNEL_MESSAGES {
                return Err(LiveError::new(
                    LiveErrorKind::Upstream,
                    "Codex live data channel pending queue is full",
                ));
            }
            pending.push_back(message);
        }
        self.flush()
    }

    pub fn flush(&self) -> Result<(), LiveError> {
        let destination = self
            .destination
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone();
        let Some(destination) = destination else {
            return Ok(());
        };
        let mut pending = self
            .pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        while let Some(message) = pending.front() {
            if destination
                .buffered_amount()
                .saturating_add(message.data.len() as u64)
                > MAX_DATA_CHANNEL_BUFFERED_AMOUNT
            {
                break;
            }
            destination.send(message)?;
            pending.pop_front();
        }
        Ok(())
    }

    pub fn close(&self) {
        if self.closed.swap(true, Ordering::AcqRel) {
            return;
        }
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clear();
        if let Some(destination) = self
            .destination
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
        {
            destination.close();
        }
    }

    #[must_use]
    pub fn pending_len(&self) -> usize {
        self.pending
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
    }
}

impl Default for DataChannelPipe {
    fn default() -> Self {
        Self::new()
    }
}

#[must_use]
pub fn is_public_remote_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(ip) => {
            if ip.is_unspecified()
                || ip.is_loopback()
                || ip.is_private()
                || ip.is_link_local()
                || ip.is_multicast()
                || ip.is_broadcast()
            {
                return false;
            }
            let octets = ip.octets();
            !(octets[0] == 100 && (64..=127).contains(&octets[1])
                || octets[0] == 192 && octets[1] == 0
                || octets[0] == 198 && (18..=19).contains(&octets[1])
                || octets[0] == 198 && octets[1] == 51 && octets[2] == 100
                || octets[0] == 203 && octets[1] == 0 && octets[2] == 113
                || octets[0] >= 240)
        }
        IpAddr::V6(ip) => {
            if ip.is_unspecified() || ip.is_loopback() || ip.is_multicast() {
                return false;
            }
            let segments = ip.segments();
            !((segments[0] & 0xfe00) == 0xfc00
                || (segments[0] & 0xffc0) == 0xfe80
                || segments[0] == 0x2001 && segments[1] == 0x0db8
                || segments[0] == 0x2002)
        }
    }
}

/// Pion clears RTP padding before forwarding; this helper captures the same
/// wire-visible normalization without binding the core port to a media crate.
pub fn normalize_rtp_padding(padding: &mut bool, padding_size: &mut u8) {
    *padding = false;
    *padding_size = 0;
}

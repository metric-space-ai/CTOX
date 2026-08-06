// ref: internal/client/codex/live/media_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use super::*;

#[test]
fn media_route_selects_remote_proxy_mode_without_leaking_credentials() {
    let route = MediaRoute {
        proxy_url: "socks5://user:password@proxy.example:1080".to_owned(),
        credential: "account-a".to_owned(),
        auth_index: "index-a".to_owned(),
    };
    assert_eq!(route.proxy_scheme(), "socks5");
    let diagnostic = route.diagnostic();
    assert!(diagnostic.contains("proxy=socks5"));
    assert!(!diagnostic.contains("password"));
    assert!(!diagnostic.contains("proxy.example"));
}

#[test]
fn media_session_limiter_is_shared_across_reconfiguration() {
    let limiter = Arc::new(MediaSessionLimiter::new(2));
    let first = limiter.acquire().unwrap();
    let second = limiter.acquire().unwrap();
    assert!(limiter.acquire().is_none());
    limiter.set_limit(3);
    let third = limiter.acquire().unwrap();
    assert_eq!(limiter.active(), 3);
    drop((first, second, third));
    assert_eq!(limiter.active(), 0);
}

struct Sink {
    messages: Mutex<Vec<DataChannelMessage>>,
    buffered: AtomicU64,
    closed: AtomicBool,
}

impl DataChannelSink for Sink {
    fn buffered_amount(&self) -> u64 {
        self.buffered.load(Ordering::Acquire)
    }

    fn send(&self, message: &DataChannelMessage) -> Result<(), LiveError> {
        self.messages.lock().unwrap().push(message.clone());
        Ok(())
    }

    fn close(&self) {
        self.closed.store(true, Ordering::Release);
    }
}

#[test]
fn data_channel_bridge_buffers_then_preserves_order() {
    let pipe = DataChannelPipe::new();
    pipe.push(DataChannelMessage {
        binary: false,
        data: b"one".to_vec(),
    })
    .unwrap();
    pipe.push(DataChannelMessage {
        binary: true,
        data: b"two".to_vec(),
    })
    .unwrap();
    assert_eq!(pipe.pending_len(), 2);
    let sink = Arc::new(Sink {
        messages: Mutex::new(Vec::new()),
        buffered: AtomicU64::new(0),
        closed: AtomicBool::new(false),
    });
    pipe.set_destination(sink.clone()).unwrap();
    assert_eq!(
        sink.messages
            .lock()
            .unwrap()
            .iter()
            .map(|m| m.data.clone())
            .collect::<Vec<_>>(),
        vec![b"one".to_vec(), b"two".to_vec()]
    );
    pipe.close();
    assert!(sink.closed.load(Ordering::Acquire));
}

#[test]
fn data_channel_bridge_enforces_pending_bound() {
    let pipe = DataChannelPipe::new();
    for _ in 0..MAX_PENDING_DATA_CHANNEL_MESSAGES {
        pipe.push(DataChannelMessage {
            binary: false,
            data: vec![1],
        })
        .unwrap();
    }
    assert!(pipe
        .push(DataChannelMessage {
            binary: false,
            data: vec![2]
        })
        .is_err());
}

#[test]
fn public_remote_ip_filter_matches_security_boundary() {
    for ip in [
        IpAddr::V4(Ipv4Addr::LOCALHOST),
        IpAddr::V4(Ipv4Addr::new(10, 0, 0, 1)),
        IpAddr::V4(Ipv4Addr::new(192, 0, 2, 1)),
        IpAddr::V6(Ipv6Addr::LOCALHOST),
        "2001:db8::1".parse().unwrap(),
    ] {
        assert!(!is_public_remote_ip(ip), "{ip}");
    }
    assert!(is_public_remote_ip("8.8.8.8".parse().unwrap()));
    assert!(is_public_remote_ip("2606:4700:4700::1111".parse().unwrap()));
}

#[test]
fn normalize_rtp_packet_clears_padding() {
    let mut padding = true;
    let mut size = 32;
    normalize_rtp_padding(&mut padding, &mut size);
    assert!(!padding);
    assert_eq!(size, 0);
}

// ref: internal/client/codex/live/tcp_proxy_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::io::{self, Cursor, Write};

use super::*;

const PUBLIC_TCP: &str = "1 1 TCP 2122260223 8.8.8.8 443 typ host tcptype passive";

#[test]
fn proxied_candidate_restricts_shape_and_target() {
    let plan = proxied_tcp_candidate_plan(PUBLIC_TCP).unwrap().unwrap();
    assert_eq!(plan.target.to_string(), "8.8.8.8:443");
    assert!(proxied_tcp_candidate_plan("1 1 UDP 1 8.8.8.8 443 typ host")
        .unwrap()
        .is_none());
    assert!(
        proxied_tcp_candidate_plan("1 1 TCP 1 8.8.8.8 443 typ host tcptype active")
            .unwrap()
            .is_none()
    );
    assert!(
        proxied_tcp_candidate_plan("1 2 TCP 1 8.8.8.8 443 typ host tcptype passive")
            .unwrap()
            .is_none()
    );
}

#[test]
fn proxied_candidate_rejects_unsafe_targets_and_ports() {
    for target in ["127.0.0.1", "10.0.0.1", "192.0.2.1", "::1", "2001:db8::1"] {
        let candidate = format!("1 1 TCP 1 {target} 443 typ host tcptype passive");
        assert!(proxied_tcp_candidate_plan(&candidate).is_err(), "{target}");
    }
    assert!(proxied_tcp_candidate_plan("1 1 TCP 1 8.8.8.8 80 typ host tcptype passive").is_err());
}

#[test]
fn rewrite_answer_filters_and_rewrites_candidates() {
    let answer = format!(
        "v=0\r\na=candidate:{PUBLIC_TCP}\r\na=candidate:2 1 UDP 1 8.8.4.4 3478 typ host\r\n"
    );
    let (rewritten, targets) =
        rewrite_proxied_upstream_answer(&answer, &["127.0.0.1:32000".parse().unwrap()]).unwrap();
    assert!(rewritten.contains("127.0.0.1 32000"));
    assert!(!rewritten.contains("UDP"));
    assert_eq!(targets, vec!["8.8.8.8:443".parse().unwrap()]);
}

#[test]
fn rewrite_answer_limits_candidate_count() {
    let candidates = (0..=MAX_UPSTREAM_ICE_CANDIDATES)
        .map(|index| format!("a=candidate:{index} 1 UDP 1 8.8.8.8 1 typ host"))
        .collect::<Vec<_>>()
        .join("\r\n");
    assert!(rewrite_proxied_upstream_answer(&candidates, &[]).is_err());
}

#[test]
fn bundled_ice_credentials_accepts_session_and_rejects_mixed_media() {
    let session = "v=0\r\na=ice-ufrag:u\r\na=ice-pwd:p\r\nm=audio 9 UDP/TLS/RTP/SAVPF 111\r\n";
    assert_eq!(
        bundled_ice_credentials(session).unwrap(),
        IceCredentials {
            ufrag: "u".to_owned(),
            password: "p".to_owned()
        }
    );
    let mixed = "v=0\r\nm=audio 9 x 1\r\na=ice-ufrag:a\r\na=ice-pwd:p\r\nm=video 9 x 1\r\na=ice-ufrag:b\r\na=ice-pwd:p\r\n";
    assert!(bundled_ice_credentials(mixed).is_err());
}

fn stun_frame(username: &str, integrity: bool, fingerprint: bool) -> Vec<u8> {
    let mut attributes = Vec::new();
    add_attribute(&mut attributes, 0x0006, username.as_bytes());
    if integrity {
        add_attribute(&mut attributes, 0x0008, &[0; 20]);
    }
    if fingerprint {
        add_attribute(&mut attributes, 0x8028, &[0; 4]);
    }
    let mut payload = vec![0x00, 0x01];
    payload.extend_from_slice(&(attributes.len() as u16).to_be_bytes());
    payload.extend_from_slice(&[0x21, 0x12, 0xA4, 0x42]);
    payload.extend_from_slice(&[0; 12]);
    payload.extend_from_slice(&attributes);
    let mut frame = (payload.len() as u16).to_be_bytes().to_vec();
    frame.extend_from_slice(&payload);
    frame
}

fn add_attribute(output: &mut Vec<u8>, kind: u16, value: &[u8]) {
    output.extend_from_slice(&kind.to_be_bytes());
    output.extend_from_slice(&(value.len() as u16).to_be_bytes());
    output.extend_from_slice(value);
    while !output.len().is_multiple_of(4) {
        output.push(0);
    }
}

#[test]
fn validated_binding_frame_checks_username_integrity_and_fingerprint() {
    let frame = stun_frame("remote:local", true, true);
    let parsed = read_validated_ice_binding_frame(
        &mut Cursor::new(frame.clone()),
        "remote:local",
        "password",
        &|_, password| password == "password",
    )
    .unwrap();
    assert_eq!(parsed.raw, frame);
    assert_eq!(parsed.username, "remote:local");
    assert!(read_validated_ice_binding_frame(
        &mut Cursor::new(stun_frame("wrong", true, true)),
        "remote:local",
        "password",
        &|_, _| true,
    )
    .is_err());
}

struct ShortWriter(Vec<u8>);

impl Write for ShortWriter {
    fn write(&mut self, data: &[u8]) -> io::Result<usize> {
        let count = data.len().min(2);
        self.0.extend_from_slice(&data[..count]);
        Ok(count)
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

#[test]
fn write_all_handles_partial_writes() {
    let mut writer = ShortWriter(Vec::new());
    write_all(&mut writer, b"abcdef").unwrap();
    assert_eq!(writer.0, b"abcdef");
}

#[test]
fn proxy_scheme_is_safe_summary() {
    assert_eq!(proxy_scheme("SOCKS5://user:pass@host"), "socks5");
    assert_eq!(proxy_scheme("not-a-url"), "proxy");
}

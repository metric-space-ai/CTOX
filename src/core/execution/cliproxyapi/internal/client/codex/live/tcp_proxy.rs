// ref: internal/client/codex/live/tcp_proxy.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::io::{self, Read, Write};
use std::net::{IpAddr, SocketAddr};

use super::{LiveError, LiveErrorKind};

pub const MAX_UPSTREAM_ICE_CANDIDATES: usize = 64;
pub const MAX_PROXIED_TCP_CANDIDATES: usize = 16;
pub const MAX_UNAUTHENTICATED_TCP_CONNECTIONS: usize = 4;
pub const MAX_INITIAL_STUN_FRAME_SIZE: usize = 4096;
pub const STUN_MESSAGE_HEADER_SIZE: usize = 20;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct IceCredentials {
    pub ufrag: String,
    pub password: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TcpCandidatePlan {
    pub fields: Vec<String>,
    pub target: SocketAddr,
}

/// Parses the candidate grammar used by Pion and applies the upstream SSRF
/// boundary: only public passive host TCP/RTP candidates on port 443 survive.
pub fn proxied_tcp_candidate_plan(
    raw_candidate: &str,
) -> Result<Option<TcpCandidatePlan>, LiveError> {
    let fields: Vec<_> = raw_candidate
        .split_whitespace()
        .map(str::to_owned)
        .collect();
    if fields.len() < 8 {
        return Err(upstream("upstream WebRTC TCP proxy candidate is malformed"));
    }
    let transport = fields[2].to_ascii_lowercase();
    if transport != "tcp" {
        return Ok(None);
    }
    let component: u16 = fields[1]
        .parse()
        .map_err(|_| upstream("parse upstream WebRTC candidate component"))?;
    let port: u16 = fields[5]
        .parse()
        .map_err(|_| upstream("parse upstream WebRTC candidate port"))?;
    let candidate_type = extension_value(&fields, "typ").unwrap_or_default();
    let tcp_type = extension_value(&fields, "tcptype").unwrap_or_default();
    if component != 1 || !candidate_type.eq_ignore_ascii_case("host") {
        return Ok(None);
    }
    if !tcp_type.eq_ignore_ascii_case("passive") {
        return Ok(None);
    }
    if port != 443 {
        return Err(upstream(format!(
            "upstream WebRTC TCP proxy candidate uses disallowed port {port}"
        )));
    }
    let address: IpAddr = fields[4]
        .parse()
        .map_err(|_| upstream("upstream WebRTC TCP proxy candidate address must be an IP"))?;
    if !is_public_proxy_target(address) {
        return Err(upstream(
            "upstream WebRTC TCP proxy candidate address must be globally routable",
        ));
    }
    Ok(Some(TcpCandidatePlan {
        fields,
        target: SocketAddr::new(address, port),
    }))
}

fn extension_value<'a>(fields: &'a [String], name: &str) -> Option<&'a str> {
    fields.windows(2).find_map(|pair| {
        pair[0]
            .eq_ignore_ascii_case(name)
            .then_some(pair[1].as_str())
    })
}

#[must_use]
pub fn is_public_proxy_target(address: IpAddr) -> bool {
    super::is_public_remote_ip(address)
        && match address {
            IpAddr::V4(ip) => {
                let [a, b, c, _] = ip.octets();
                !(a == 0 || (a == 192 && b == 88 && c == 99) || (a == 224) || a >= 240)
            }
            IpAddr::V6(ip) => {
                let segments = ip.segments();
                !(segments[0] == 0
                    || segments[0] == 0x0064 && segments[1] == 0xff9b
                    || segments[0] == 0x2001 && segments[1] <= 0x01ff
                    || segments[0] == 0x3fff
                    || segments[0] == 0x5f00)
            }
        }
}

/// Reads consistent bundle-level ICE credentials from session/media SDP.
pub fn bundled_ice_credentials(sdp: &str) -> Result<IceCredentials, LiveError> {
    let mut session = IceCredentials {
        ufrag: String::new(),
        password: String::new(),
    };
    let mut current = session.clone();
    let mut selected: Option<IceCredentials> = None;
    let mut in_media = false;
    for raw in sdp.lines() {
        let line = raw.trim_end_matches('\r').trim();
        if line.starts_with("m=") {
            if in_media {
                select_credentials(&current, &mut selected)?;
            }
            in_media = true;
            current = session.clone();
        } else if let Some(value) = line.strip_prefix("a=ice-ufrag:") {
            if in_media {
                current.ufrag = value.trim().to_owned();
            } else {
                session.ufrag = value.trim().to_owned();
                current.ufrag = session.ufrag.clone();
            }
        } else if let Some(value) = line.strip_prefix("a=ice-pwd:") {
            if in_media {
                current.password = value.trim().to_owned();
            } else {
                session.password = value.trim().to_owned();
                current.password = session.password.clone();
            }
        }
    }
    if in_media {
        select_credentials(&current, &mut selected)?;
    }
    let selected = selected.unwrap_or(session);
    if selected.ufrag.is_empty() || selected.password.is_empty() {
        return Err(upstream("SDP is missing ICE credentials"));
    }
    Ok(selected)
}

fn select_credentials(
    credentials: &IceCredentials,
    selected: &mut Option<IceCredentials>,
) -> Result<(), LiveError> {
    if credentials.ufrag.is_empty() && credentials.password.is_empty() {
        return Ok(());
    }
    if credentials.ufrag.is_empty() || credentials.password.is_empty() {
        return Err(upstream("SDP contains incomplete ICE credentials"));
    }
    if selected
        .as_ref()
        .is_some_and(|selected| selected != credentials)
    {
        return Err(upstream(
            "SDP contains inconsistent bundled ICE credentials",
        ));
    }
    *selected = Some(credentials.clone());
    Ok(())
}

/// Deterministically filters SDP candidates and replaces accepted addresses
/// with caller-provided loopback listener addresses. Actual dialing/listening
/// remains explicit host authority.
pub fn rewrite_proxied_upstream_answer(
    answer: &str,
    listeners: &[SocketAddr],
) -> Result<(String, Vec<SocketAddr>), LiveError> {
    let mut candidate_count = 0;
    let mut proxy_count = 0;
    let mut targets = Vec::new();
    let mut output = Vec::new();
    for raw in answer.lines() {
        let line = raw.trim_end_matches('\r');
        let candidate = line
            .strip_prefix("a=candidate:")
            .or_else(|| line.strip_prefix("candidate:"));
        let Some(candidate) = candidate else {
            output.push(line.to_owned());
            continue;
        };
        candidate_count += 1;
        if candidate_count > MAX_UPSTREAM_ICE_CANDIDATES {
            return Err(upstream(format!(
                "upstream WebRTC answer exceeds the {MAX_UPSTREAM_ICE_CANDIDATES} candidate limit"
            )));
        }
        let Some(plan) = proxied_tcp_candidate_plan(candidate)? else {
            continue;
        };
        if proxy_count >= MAX_PROXIED_TCP_CANDIDATES {
            return Err(upstream(format!(
                "upstream WebRTC answer exceeds the {MAX_PROXIED_TCP_CANDIDATES} TCP candidate proxy limit"
            )));
        }
        let listener = listeners
            .get(proxy_count)
            .ok_or_else(|| upstream("Codex live TCP proxy listener returned an invalid address"))?;
        if !listener.ip().is_loopback() {
            return Err(upstream("Codex live TCP proxy listener must be loopback"));
        }
        let mut fields = plan.fields;
        fields[4] = listener.ip().to_string();
        fields[5] = listener.port().to_string();
        output.push(format!("a=candidate:{}", fields.join(" ")));
        targets.push(plan.target);
        proxy_count += 1;
    }
    if targets.is_empty() {
        return Err(upstream(
            "upstream WebRTC answer has no supported public TCP passive candidate on port 443",
        ));
    }
    let newline = if answer.contains("\r\n") {
        "\r\n"
    } else {
        "\n"
    };
    let mut rewritten = output.join(newline);
    if answer.ends_with(newline) {
        rewritten.push_str(newline);
    }
    Ok((rewritten, targets))
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StunBindingFrame {
    pub raw: Vec<u8>,
    pub username: String,
    pub has_message_integrity: bool,
    pub has_fingerprint: bool,
}

/// Parses and bounds the first RFC 4571 framed STUN binding request. Integrity
/// cryptography is delegated to an injected verifier to avoid hidden authority.
pub fn read_validated_ice_binding_frame(
    reader: &mut dyn Read,
    expected_user: &str,
    remote_password: &str,
    verify: &dyn Fn(&[u8], &str) -> bool,
) -> Result<StunBindingFrame, LiveError> {
    let mut header = [0_u8; 2];
    reader
        .read_exact(&mut header)
        .map_err(|error| upstream(format!("read ICE-TCP frame header: {error}")))?;
    let size = u16::from_be_bytes(header) as usize;
    if !(STUN_MESSAGE_HEADER_SIZE..=MAX_INITIAL_STUN_FRAME_SIZE).contains(&size) {
        return Err(upstream(format!(
            "invalid initial ICE-TCP STUN frame size {size}"
        )));
    }
    let mut payload = vec![0; size];
    reader
        .read_exact(&mut payload)
        .map_err(|error| upstream(format!("read ICE-TCP STUN frame: {error}")))?;
    if payload[0..2] != [0x00, 0x01] {
        return Err(upstream("initial ICE-TCP STUN message has unexpected type"));
    }
    let declared = u16::from_be_bytes([payload[2], payload[3]]) as usize;
    if declared + STUN_MESSAGE_HEADER_SIZE != size {
        return Err(upstream(
            "initial ICE-TCP STUN message contains trailing data",
        ));
    }
    if payload[4..8] != [0x21, 0x12, 0xA4, 0x42] {
        return Err(upstream("decode initial ICE-TCP STUN message"));
    }
    let (username, has_integrity, has_fingerprint) = parse_stun_attributes(&payload)?;
    if username != expected_user {
        return Err(upstream(
            "initial ICE-TCP STUN username does not match the media session",
        ));
    }
    if !has_integrity || !verify(&payload, remote_password) {
        return Err(upstream(
            "verify initial ICE-TCP STUN integrity: verification failed",
        ));
    }
    if !has_fingerprint {
        return Err(upstream(
            "verify initial ICE-TCP STUN fingerprint: missing fingerprint",
        ));
    }
    let mut raw = header.to_vec();
    raw.extend_from_slice(&payload);
    Ok(StunBindingFrame {
        raw,
        username,
        has_message_integrity: has_integrity,
        has_fingerprint,
    })
}

fn parse_stun_attributes(payload: &[u8]) -> Result<(String, bool, bool), LiveError> {
    let mut offset = STUN_MESSAGE_HEADER_SIZE;
    let mut username = None;
    let mut integrity = false;
    let mut fingerprint = false;
    while offset < payload.len() {
        if offset + 4 > payload.len() {
            return Err(upstream("decode initial ICE-TCP STUN message"));
        }
        let kind = u16::from_be_bytes([payload[offset], payload[offset + 1]]);
        let length = u16::from_be_bytes([payload[offset + 2], payload[offset + 3]]) as usize;
        offset += 4;
        if offset + length > payload.len() {
            return Err(upstream("decode initial ICE-TCP STUN message"));
        }
        match kind {
            0x0006 => {
                username = Some(
                    String::from_utf8(payload[offset..offset + length].to_vec())
                        .map_err(|_| upstream("read initial ICE-TCP STUN username"))?,
                );
            }
            0x0008 => integrity = true,
            0x8028 => fingerprint = true,
            _ => {}
        }
        offset += (length + 3) & !3;
    }
    Ok((
        username.ok_or_else(|| upstream("read initial ICE-TCP STUN username"))?,
        integrity,
        fingerprint,
    ))
}

pub fn write_all(writer: &mut dyn Write, mut data: &[u8]) -> io::Result<()> {
    while !data.is_empty() {
        let written = writer.write(data)?;
        if written == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        data = &data[written..];
    }
    Ok(())
}

#[must_use]
pub fn proxy_scheme(raw_proxy_url: &str) -> String {
    raw_proxy_url
        .trim()
        .split_once("://")
        .map(|(scheme, _)| scheme.to_ascii_lowercase())
        .filter(|scheme| !scheme.is_empty())
        .unwrap_or_else(|| "proxy".to_owned())
}

fn upstream(message: impl Into<String>) -> LiveError {
    LiveError::new(LiveErrorKind::Upstream, message)
}

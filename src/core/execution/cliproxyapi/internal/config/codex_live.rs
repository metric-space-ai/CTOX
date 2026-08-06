// ref: internal/config/codex_live.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::net::IpAddr;

use serde::{Deserialize, Deserializer, Serialize};
use url::Url;

pub const DEFAULT_CODEX_LIVE_MEDIA_MAX_SESSIONS: i32 = 32;

#[derive(Clone, Debug, Default, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct CodexLiveIceServer {
    #[serde(default)]
    pub urls: Vec<String>,
    #[serde(default, skip_serializing)]
    pub username: String,
    #[serde(default, skip_serializing)]
    pub credential: String,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct CodexLiveMediaRelayConfig {
    pub enabled: bool,
    pub max_sessions: i32,
    pub disable_private_remote_ips: bool,
    pub public_ip: String,
    pub udp_port_min: u16,
    pub udp_port_max: u16,
    pub ice_servers: Vec<CodexLiveIceServer>,
}

#[derive(Deserialize)]
#[serde(rename_all = "kebab-case")]
struct RelayWire {
    #[serde(default)]
    enabled: bool,
    #[serde(default)]
    max_sessions: i32,
    #[serde(default)]
    disable_private_remote_ips: Option<bool>,
    #[serde(default)]
    allow_private_remote_ips: Option<bool>,
    #[serde(default)]
    public_ip: String,
    #[serde(default)]
    udp_port_min: u16,
    #[serde(default)]
    udp_port_max: u16,
    #[serde(default)]
    ice_servers: Vec<CodexLiveIceServer>,
}

impl<'de> Deserialize<'de> for CodexLiveMediaRelayConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = RelayWire::deserialize(deserializer)?;
        if wire.allow_private_remote_ips.is_some() && wire.disable_private_remote_ips.is_some() {
            return Err(serde::de::Error::custom(
                "codex.live-media-relay cannot set both allow-private-remote-ips and disable-private-remote-ips",
            ));
        }
        let disable_private_remote_ips = wire
            .allow_private_remote_ips
            .map(|allow| !allow)
            .or(wire.disable_private_remote_ips)
            .unwrap_or(false);
        Ok(Self {
            enabled: wire.enabled,
            max_sessions: wire.max_sessions,
            disable_private_remote_ips,
            public_ip: wire.public_ip,
            udp_port_min: wire.udp_port_min,
            udp_port_max: wire.udp_port_max,
            ice_servers: wire.ice_servers,
        })
    }
}

impl CodexLiveMediaRelayConfig {
    #[must_use]
    pub fn effective_max_sessions(&self) -> i32 {
        if self.max_sessions > 0 {
            self.max_sessions
        } else {
            DEFAULT_CODEX_LIVE_MEDIA_MAX_SESSIONS
        }
    }

    pub fn validate(&self) -> Result<(), CodexLiveConfigError> {
        if !self.enabled {
            return Ok(());
        }
        if self.max_sessions < 0 {
            return Err(CodexLiveConfigError::new(
                "codex.live-media-relay.max-sessions must not be negative",
            ));
        }
        let public_ip = self.public_ip.trim();
        if !public_ip.is_empty() && public_ip.parse::<IpAddr>().is_err() {
            return Err(CodexLiveConfigError::new(format!(
                "codex.live-media-relay.public-ip is invalid: {public_ip:?}"
            )));
        }
        if (self.udp_port_min == 0) != (self.udp_port_max == 0) {
            return Err(CodexLiveConfigError::new(
                "codex.live-media-relay UDP port minimum and maximum must both be set",
            ));
        }
        if self.udp_port_min > self.udp_port_max {
            return Err(CodexLiveConfigError::new(
                "codex.live-media-relay.udp-port-min must not exceed udp-port-max",
            ));
        }
        if self.udp_port_min != 0 {
            let available = i32::from(self.udp_port_max) - i32::from(self.udp_port_min) + 1;
            let required = self.effective_max_sessions() * 2;
            if available < required {
                return Err(CodexLiveConfigError::new(format!(
                    "codex.live-media-relay UDP range requires at least {required} ports for {} sessions",
                    self.effective_max_sessions()
                )));
            }
        }
        for (server_index, server) in self.ice_servers.iter().enumerate() {
            if server.urls.is_empty() {
                return Err(CodexLiveConfigError::new(format!(
                    "codex.live-media-relay.ice-servers[{server_index}].urls is required"
                )));
            }
            for raw_url in &server.urls {
                let parsed = Url::parse(raw_url.trim()).map_err(|_| {
                    CodexLiveConfigError::new(format!(
                        "codex.live-media-relay.ice-servers[{server_index}] contains an invalid URL"
                    ))
                })?;
                if !matches!(
                    parsed.scheme().to_ascii_lowercase().as_str(),
                    "stun" | "stuns" | "turn" | "turns"
                ) {
                    return Err(CodexLiveConfigError::new(format!(
                        "codex.live-media-relay.ice-servers[{server_index}] uses unsupported scheme {:?}",
                        parsed.scheme()
                    )));
                }
            }
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CodexLiveConfigError(String);

impl CodexLiveConfigError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CodexLiveConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CodexLiveConfigError {}

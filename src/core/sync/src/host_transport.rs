//! Host-owned transport settings kept in the encrypted secret store, separately
//! from immutable public identity pins. Business OS credentials are never reused.
use crate::host_config::{HostConfiguration, HostMember};
use rxdb::plugins::replication_webrtc::RTCIceServer;
use std::io;

pub use crate::contracts::{SyncHostIceServer as IceServer, SyncHostTransport as HostTransport};
fn invalid() -> io::Error {
    // Never include credential-bearing URLs or input JSON in a diagnostic.
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "invalid execution-host transport settings",
    )
}
impl HostTransport {
    pub fn parse(input: &str, config: &HostConfiguration) -> io::Result<Self> {
        let transport: Self = serde_json::from_str(input).map_err(|_| invalid())?;
        transport.validate(config)?;
        Ok(transport)
    }
    pub fn validate(&self, config: &HostConfiguration) -> io::Result<()> {
        config.validate()?;
        let role = match config.local {
            HostMember::Voter { .. } => "ctox_instance",
            HostMember::Worker { .. } => "workjet_executor",
        };
        if self.signaling_urls.is_empty()
            || self.signaling_urls.len() > 8
            || self.ice_servers.len() > 8
        {
            return Err(invalid());
        }
        for raw in &self.signaling_urls {
            if raw.len() > 16384 {
                return Err(invalid());
            }
            let url = url::Url::parse(raw).map_err(|_| invalid())?;
            let loopback = matches!(url.host_str(), Some("127.0.0.1" | "localhost" | "[::1]"));
            if !(url.scheme() == "wss" || (url.scheme() == "ws" && loopback))
                || !url.username().is_empty()
                || url.password().is_some()
                || url.fragment().is_some()
            {
                return Err(invalid());
            }
            let pairs: Vec<_> = url.query_pairs().collect();
            if pairs.iter().filter(|(key, _)| key == "role").count() != 1
                || !pairs
                    .iter()
                    .any(|(key, value)| key == "role" && value == role)
                || pairs.iter().any(|(key, value)| {
                    matches!(
                        key.as_ref(),
                        "browser_token_hash" | "native_token_hash" | "peer_role"
                    ) || (key == "auth_version" && value == "ctox-role-bound-v1")
                        || (key == "instance_id" && value != config.scope_id.as_str())
                })
            {
                return Err(invalid());
            }
        }
        for ice in &self.ice_servers {
            if ice.urls.is_empty()
                || ice.urls.len() > 8
                || ice.username.len() > 4096
                || ice.credential.len() > 4096
                || ice.urls.iter().any(|url| {
                    url.len() > 4096
                        || !["stun:", "stuns:", "turn:", "turns:"]
                            .iter()
                            .any(|prefix| url.starts_with(prefix))
                })
            {
                return Err(invalid());
            }
        }
        Ok(())
    }
    pub fn native_ice_servers(&self) -> Vec<RTCIceServer> {
        // An explicit empty host configuration must not silently choose public
        // STUN servers. Signaling may still supply its authenticated ICE bootstrap.
        if self.ice_servers.is_empty() {
            return vec![RTCIceServer::default()];
        }
        self.ice_servers
            .iter()
            .map(|ice| RTCIceServer {
                urls: ice.urls.clone(),
                username: ice.username.clone(),
                credential: ice.credential.clone(),
            })
            .collect()
    }
}

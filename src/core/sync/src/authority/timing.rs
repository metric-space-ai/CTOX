//! Host-owned timing for Internet-connected authority peers.
use serde::{Deserialize, Serialize};
use std::io;

/// OpenRaft 0.9 uses heartbeat_interval as the complete quorum-read RPC budget,
/// not only the interval between heartbeats. Its 50 ms default targets low-latency
/// datacenter networks. Keep this policy explicit at the native host boundary.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub struct AuthorityTiming {
    pub heartbeat_ms: u64,
    pub election_min_ms: u64,
    pub election_max_ms: u64,
}

impl Default for AuthorityTiming {
    fn default() -> Self {
        Self {
            heartbeat_ms: 250,
            election_min_ms: 1_500,
            election_max_ms: 3_000,
        }
    }
}

impl AuthorityTiming {
    pub fn raft_config(self) -> io::Result<openraft::Config> {
        if self.heartbeat_ms == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "authority heartbeat must be positive",
            ));
        }
        openraft::Config {
            cluster_name: "ctox-sync-authority".into(),
            heartbeat_interval: self.heartbeat_ms,
            election_timeout_min: self.election_min_ms,
            election_timeout_max: self.election_max_ms,
            ..Default::default()
        }
        .validate()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error.to_string()))
    }
}

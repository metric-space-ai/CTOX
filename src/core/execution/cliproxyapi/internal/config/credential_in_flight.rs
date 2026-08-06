// ref: internal/config/credential_in_flight.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only
use serde::{Deserialize, Serialize};
use std::time::Duration;
pub const DEFAULT_IN_FLIGHT_MAX_PART_BYTES: usize = 256 * 1024;
pub const DEFAULT_IN_FLIGHT_MAX_PART_COUNT: usize = 64;
pub const DEFAULT_IN_FLIGHT_MAX_REVISION_BYTES: usize = 16 * 1024 * 1024;
pub const DEFAULT_IN_FLIGHT_MAX_AGGREGATE_GROUPS: usize = 100_000;
pub const DEFAULT_IN_FLIGHT_MAX_DETAILS: usize = 10_000;
pub const DEFAULT_IN_FLIGHT_MAX_STRING_BYTES: usize = 256;
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "kebab-case")]
pub struct CredentialInFlightConfig {
    pub snapshot_interval: String,
    pub stale_after: String,
    pub max_part_bytes: usize,
    pub max_part_count: usize,
    pub max_revision_bytes: usize,
    pub max_aggregate_groups: usize,
    pub max_details: usize,
    pub max_string_bytes: usize,
    pub staging_retention: String,
}
impl Default for CredentialInFlightConfig {
    fn default() -> Self {
        Self {
            snapshot_interval: "2s".into(),
            stale_after: "10s".into(),
            max_part_bytes: DEFAULT_IN_FLIGHT_MAX_PART_BYTES,
            max_part_count: DEFAULT_IN_FLIGHT_MAX_PART_COUNT,
            max_revision_bytes: DEFAULT_IN_FLIGHT_MAX_REVISION_BYTES,
            max_aggregate_groups: DEFAULT_IN_FLIGHT_MAX_AGGREGATE_GROUPS,
            max_details: DEFAULT_IN_FLIGHT_MAX_DETAILS,
            max_string_bytes: DEFAULT_IN_FLIGHT_MAX_STRING_BYTES,
            staging_retention: "1m".into(),
        }
    }
}
impl CredentialInFlightConfig {
    pub fn durations(&self) -> Result<(Duration, Duration, Duration), String> {
        let snapshot = parse_duration(&self.snapshot_interval)
            .ok_or("credential-in-flight.snapshot-interval must be positive")?;
        let stale = parse_duration(&self.stale_after)
            .ok_or("credential-in-flight.stale-after must be at least three snapshot intervals")?;
        if snapshot > stale / 3 {
            return Err(
                "credential-in-flight.stale-after must be at least three snapshot intervals".into(),
            );
        }
        let retention = parse_duration(&self.staging_retention)
            .ok_or("credential-in-flight.staging-retention must be positive")?;
        Ok((snapshot, stale, retention))
    }
    pub fn validate(&self) -> Result<(), String> {
        self.durations()?;
        if self.max_part_bytes < 1024 || self.max_part_count == 0 || self.max_part_count > 64 {
            return Err("credential-in-flight part bounds are invalid".into());
        }
        if self.max_revision_bytes < self.max_part_bytes
            || self.max_revision_bytes > DEFAULT_IN_FLIGHT_MAX_REVISION_BYTES
        {
            return Err("credential-in-flight.max-revision-bytes is outside hard bounds".into());
        }
        if self.max_revision_bytes.div_ceil(self.max_part_bytes) > self.max_part_count {
            return Err("credential-in-flight.max-revision-bytes exceeds part capacity".into());
        }
        if self.max_aggregate_groups == 0
            || self.max_aggregate_groups > DEFAULT_IN_FLIGHT_MAX_AGGREGATE_GROUPS
            || self.max_details > DEFAULT_IN_FLIGHT_MAX_DETAILS
            || self.max_string_bytes == 0
            || self.max_string_bytes > DEFAULT_IN_FLIGHT_MAX_STRING_BYTES
        {
            return Err("credential-in-flight bounds are invalid".into());
        }
        Ok(())
    }
}
fn parse_duration(raw: &str) -> Option<Duration> {
    let raw = raw.trim();
    let (number, multiplier) = if let Some(v) = raw.strip_suffix("ms") {
        (v, Duration::from_millis(1))
    } else if let Some(v) = raw.strip_suffix('s') {
        (v, Duration::from_secs(1))
    } else if let Some(v) = raw.strip_suffix('m') {
        (v, Duration::from_secs(60))
    } else {
        return None;
    };
    let count = number.parse::<u32>().ok()?;
    (count > 0).then(|| multiplier * count)
}

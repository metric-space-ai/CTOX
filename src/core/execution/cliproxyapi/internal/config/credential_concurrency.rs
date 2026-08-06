// ref: internal/config/credential_concurrency.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::time::Duration;

use serde::{Deserialize, Deserializer, Serialize};

const DEFAULT_CPA_HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(3);
const DEFAULT_CPA_CANCEL_BOUND: Duration = Duration::from_secs(5);
const DEFAULT_RECLAIM_GRACE: Duration = Duration::from_secs(5);
const DEFAULT_CLEANUP_INTERVAL: Duration = Duration::from_secs(5);
const DEFAULT_RELEASE_FLUSH_INTERVAL: Duration = Duration::from_millis(250);
const DEFAULT_RELEASE_MAX_BACKOFF: Duration = Duration::from_secs(2);
const DEFAULT_BUSY_RETRY_MIN: Duration = Duration::from_millis(250);
const DEFAULT_BUSY_RETRY_MAX: Duration = Duration::from_secs(1);
pub const MAX_CREDENTIAL_CONCURRENCY_LIMIT: i64 = 1_000_000;

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub struct CredentialConcurrencyConfig {
    pub lifecycle_config_revision: i64,
    pub observation_barrier_revision: i64,
    #[serde(with = "duration_nanos")]
    pub cpa_heartbeat_timeout: Duration,
    #[serde(with = "duration_nanos")]
    pub cpa_cancel_bound: Duration,
    #[serde(with = "duration_nanos")]
    pub reclaim_grace: Duration,
    #[serde(with = "duration_nanos")]
    pub cleanup_interval: Duration,
    #[serde(with = "duration_nanos")]
    pub release_flush_interval: Duration,
    #[serde(with = "duration_nanos")]
    pub release_max_backoff: Duration,
    #[serde(with = "duration_nanos")]
    pub busy_retry_min: Duration,
    #[serde(with = "duration_nanos")]
    pub busy_retry_max: Duration,
    pub max_limit: i64,
    #[serde(skip)]
    pub(crate) present: Presence,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub(crate) struct Presence {
    lifecycle_config_revision: bool,
    cpa_heartbeat_timeout: bool,
    cpa_cancel_bound: bool,
    reclaim_grace: bool,
    cleanup_interval: bool,
    release_flush_interval: bool,
    release_max_backoff: bool,
    busy_retry_min: bool,
    busy_retry_max: bool,
    max_limit: bool,
}

impl<'de> Deserialize<'de> for CredentialConcurrencyConfig {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut values = BTreeMap::<String, serde_yaml::Value>::deserialize(deserializer)?;
        let known = [
            "lifecycle-config-revision",
            "observation-barrier-revision",
            "cpa-heartbeat-timeout",
            "cpa-cancel-bound",
            "reclaim-grace",
            "cleanup-interval",
            "release-flush-interval",
            "release-max-backoff",
            "busy-retry-min",
            "busy-retry-max",
            "max-limit",
        ];
        if let Some(key) = values.keys().find(|key| !known.contains(&key.as_str())) {
            return Err(serde::de::Error::custom(format!("unknown field {key:?}")));
        }
        let present = Presence {
            lifecycle_config_revision: values.contains_key("lifecycle-config-revision"),
            cpa_heartbeat_timeout: values.contains_key("cpa-heartbeat-timeout"),
            cpa_cancel_bound: values.contains_key("cpa-cancel-bound"),
            reclaim_grace: values.contains_key("reclaim-grace"),
            cleanup_interval: values.contains_key("cleanup-interval"),
            release_flush_interval: values.contains_key("release-flush-interval"),
            release_max_backoff: values.contains_key("release-max-backoff"),
            busy_retry_min: values.contains_key("busy-retry-min"),
            busy_retry_max: values.contains_key("busy-retry-max"),
            max_limit: values.contains_key("max-limit"),
        };
        Ok(Self {
            lifecycle_config_revision: take_i64(&mut values, "lifecycle-config-revision")?,
            observation_barrier_revision: take_i64(&mut values, "observation-barrier-revision")?,
            cpa_heartbeat_timeout: take_duration(&mut values, "cpa-heartbeat-timeout")?,
            cpa_cancel_bound: take_duration(&mut values, "cpa-cancel-bound")?,
            reclaim_grace: take_duration(&mut values, "reclaim-grace")?,
            cleanup_interval: take_duration(&mut values, "cleanup-interval")?,
            release_flush_interval: take_duration(&mut values, "release-flush-interval")?,
            release_max_backoff: take_duration(&mut values, "release-max-backoff")?,
            busy_retry_min: take_duration(&mut values, "busy-retry-min")?,
            busy_retry_max: take_duration(&mut values, "busy-retry-max")?,
            max_limit: take_i64(&mut values, "max-limit")?,
            present,
        })
    }
}

impl CredentialConcurrencyConfig {
    #[must_use]
    pub fn with_defaults(mut self) -> Self {
        if !self.present.cpa_heartbeat_timeout && self.cpa_heartbeat_timeout.is_zero() {
            self.cpa_heartbeat_timeout = DEFAULT_CPA_HEARTBEAT_TIMEOUT;
        }
        if !self.present.cpa_cancel_bound && self.cpa_cancel_bound.is_zero() {
            self.cpa_cancel_bound = DEFAULT_CPA_CANCEL_BOUND;
        }
        if !self.present.reclaim_grace && self.reclaim_grace.is_zero() {
            self.reclaim_grace = DEFAULT_RECLAIM_GRACE;
        }
        if !self.present.cleanup_interval && self.cleanup_interval.is_zero() {
            self.cleanup_interval = DEFAULT_CLEANUP_INTERVAL;
        }
        if !self.present.release_flush_interval && self.release_flush_interval.is_zero() {
            self.release_flush_interval = DEFAULT_RELEASE_FLUSH_INTERVAL;
        }
        if !self.present.release_max_backoff && self.release_max_backoff.is_zero() {
            self.release_max_backoff = DEFAULT_RELEASE_MAX_BACKOFF;
        }
        if !self.present.busy_retry_min && self.busy_retry_min.is_zero() {
            self.busy_retry_min = DEFAULT_BUSY_RETRY_MIN;
        }
        if !self.present.busy_retry_max && self.busy_retry_max.is_zero() {
            self.busy_retry_max = DEFAULT_BUSY_RETRY_MAX;
        }
        if !self.present.max_limit && self.max_limit == 0 {
            self.max_limit = MAX_CREDENTIAL_CONCURRENCY_LIMIT;
        }
        self
    }
}

pub fn validate_credential_concurrency(
    config: &CredentialConcurrencyConfig,
) -> Result<(), CredentialConcurrencyError> {
    if config.lifecycle_config_revision < 0
        || (config.present.lifecycle_config_revision && config.lifecycle_config_revision == 0)
    {
        return Err(CredentialConcurrencyError::new(
            "lifecycle configuration revision must be positive when present",
        ));
    }
    if config.observation_barrier_revision < 0 {
        return Err(CredentialConcurrencyError::new(
            "observation barrier revision must not be negative",
        ));
    }
    if [
        config.cpa_heartbeat_timeout,
        config.cpa_cancel_bound,
        config.reclaim_grace,
        config.cleanup_interval,
    ]
    .iter()
    .any(Duration::is_zero)
    {
        return Err(CredentialConcurrencyError::new(
            "credential concurrency lifecycle durations must be positive",
        ));
    }
    if [
        config.release_flush_interval,
        config.release_max_backoff,
        config.busy_retry_min,
        config.busy_retry_max,
    ]
    .iter()
    .any(Duration::is_zero)
    {
        return Err(CredentialConcurrencyError::new(
            "credential concurrency limiter durations must be positive",
        ));
    }
    if config.release_max_backoff < config.release_flush_interval {
        return Err(CredentialConcurrencyError::new("credential concurrency release max backoff must not be less than release flush interval"));
    }
    if !config
        .busy_retry_min
        .subsec_nanos()
        .is_multiple_of(1_000_000)
        || !config
            .busy_retry_max
            .subsec_nanos()
            .is_multiple_of(1_000_000)
    {
        return Err(CredentialConcurrencyError::new(
            "credential concurrency busy retry durations must be whole milliseconds",
        ));
    }
    if config.busy_retry_max < config.busy_retry_min {
        return Err(CredentialConcurrencyError::new(
            "credential concurrency busy retry max must not be less than busy retry min",
        ));
    }
    if !(1..=MAX_CREDENTIAL_CONCURRENCY_LIMIT).contains(&config.max_limit) {
        return Err(CredentialConcurrencyError::new(format!("credential concurrency max limit must be between 1 and {MAX_CREDENTIAL_CONCURRENCY_LIMIT}")));
    }
    Ok(())
}

pub fn validate_credential_concurrency_lifecycle(
    node_heartbeat_timeout: Duration,
    config: &CredentialConcurrencyConfig,
) -> Result<(), CredentialConcurrencyError> {
    if node_heartbeat_timeout.is_zero() {
        return Err(CredentialConcurrencyError::new(
            "credential concurrency lifecycle durations must be positive",
        ));
    }
    validate_credential_concurrency(config)?;
    let left = go_duration_add(node_heartbeat_timeout, config.reclaim_grace)?;
    let right = go_duration_add(config.cpa_heartbeat_timeout, config.cpa_cancel_bound)?;
    if left <= right {
        return Err(CredentialConcurrencyError::new("node heartbeat timeout plus reclaim grace must exceed CPA heartbeat timeout plus cancel bound"));
    }
    Ok(())
}

fn go_duration_add(
    left: Duration,
    right: Duration,
) -> Result<Duration, CredentialConcurrencyError> {
    let nanos = left
        .as_nanos()
        .checked_add(right.as_nanos())
        .filter(|value| *value <= i64::MAX as u128)
        .ok_or_else(|| {
            CredentialConcurrencyError::new(
                "credential concurrency lifecycle timing safety invariant overflows",
            )
        })?;
    Ok(Duration::from_nanos(nanos as u64))
}

fn take_i64<E: serde::de::Error>(
    values: &mut BTreeMap<String, serde_yaml::Value>,
    key: &str,
) -> Result<i64, E> {
    match values.remove(key) {
        None | Some(serde_yaml::Value::Null) => Ok(0),
        Some(value) => serde_yaml::from_value(value).map_err(E::custom),
    }
}

fn take_duration<E: serde::de::Error>(
    values: &mut BTreeMap<String, serde_yaml::Value>,
    key: &str,
) -> Result<Duration, E> {
    match values.remove(key) {
        None | Some(serde_yaml::Value::Null) => Ok(Duration::ZERO),
        Some(serde_yaml::Value::Number(number)) => number
            .as_u64()
            .map(Duration::from_nanos)
            .ok_or_else(|| E::custom(format!("{key} must be a non-negative duration"))),
        Some(serde_yaml::Value::String(raw)) => parse_go_duration(&raw)
            .ok_or_else(|| E::custom(format!("{key} has invalid duration {raw:?}"))),
        Some(_) => Err(E::custom(format!("{key} must be a duration"))),
    }
}

fn parse_go_duration(raw: &str) -> Option<Duration> {
    let raw = raw.trim();
    for (suffix, nanos) in [
        ("ns", 1_u64),
        ("us", 1_000),
        ("µs", 1_000),
        ("ms", 1_000_000),
        ("s", 1_000_000_000),
        ("m", 60_000_000_000),
        ("h", 3_600_000_000_000),
    ] {
        if let Some(value) = raw.strip_suffix(suffix) {
            let value = value.parse::<f64>().ok()?;
            if value.is_sign_negative() || !value.is_finite() {
                return None;
            }
            let total = value * nanos as f64;
            if total > u64::MAX as f64 {
                return None;
            }
            return Some(Duration::from_nanos(total as u64));
        }
    }
    None
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CredentialConcurrencyError(String);

impl CredentialConcurrencyError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}
impl fmt::Display for CredentialConcurrencyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CredentialConcurrencyError {}

mod duration_nanos {
    use serde::Serializer;
    use std::time::Duration;
    pub fn serialize<S: Serializer>(value: &Duration, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_u128(value.as_nanos())
    }
}

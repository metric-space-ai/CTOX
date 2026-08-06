// ref: internal/api/handlers/management/handler.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

use crate::internal::config::CliproxyRuntimeConfig;
use crate::internal::pluginhost::support::support_plugin_header_value;

use super::{ManagementConfigError, ManagementConfigService};

const MAX_FAILURES: u8 = 5;
const BAN_DURATION_MS: i64 = 30 * 60 * 1_000;
const MAX_IDLE_MS: i64 = 2 * 60 * 60 * 1_000;

#[must_use]
pub fn management_support_plugin_header() -> &'static str {
    support_plugin_header_value()
}

/// Injected runtime authority notified after a management configuration was
/// durably validated and saved.
pub trait ManagementConfigReload: Send + Sync {
    fn apply(&self, generation: u64, config: &CliproxyRuntimeConfig);
}

/// CTOX adaptation of upstream's management `Handler` owner.
///
/// Persistence, runtime application and authentication time are all injected.
/// This owner coordinates the upstream save-then-reload invariant without
/// reading configuration, secrets or process environment on its own.
pub struct ManagementHandlerOwner {
    authenticator: Arc<ManagementAuthenticator>,
    config: Arc<ManagementConfigService>,
    reload: Option<Arc<dyn ManagementConfigReload>>,
    mutation: Mutex<u64>,
}

impl ManagementHandlerOwner {
    #[must_use]
    pub fn new(
        authenticator: Arc<ManagementAuthenticator>,
        config: Arc<ManagementConfigService>,
    ) -> Self {
        Self {
            authenticator,
            config,
            reload: None,
            mutation: Mutex::new(0),
        }
    }

    #[must_use]
    pub fn with_reload(mut self, reload: Arc<dyn ManagementConfigReload>) -> Self {
        self.reload = Some(reload);
        self
    }

    pub fn authenticate(
        &self,
        client_ip: &str,
        local_client: bool,
        provided: Option<&str>,
    ) -> Result<(), ManagementAuthError> {
        self.authenticator
            .authenticate(client_ip, local_client, provided)
    }

    pub fn config_snapshot(&self) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        self.config.snapshot()
    }

    pub fn replace_config(
        &self,
        config: CliproxyRuntimeConfig,
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        self.save_and_reload(|service| service.replace(config))
    }

    pub fn set_request_timeout_ms(
        &self,
        request_timeout_ms: u64,
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        self.save_and_reload(|service| service.set_request_timeout_ms(request_timeout_ms))
    }

    pub fn set_routing_strategy(
        &self,
        strategy: &str,
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        self.save_and_reload(|service| service.set_routing_strategy(strategy))
    }

    fn save_and_reload(
        &self,
        save: impl FnOnce(
            &ManagementConfigService,
        ) -> Result<CliproxyRuntimeConfig, ManagementConfigError>,
    ) -> Result<CliproxyRuntimeConfig, ManagementConfigError> {
        let mut generation = self
            .mutation
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = save(&self.config)?;
        *generation = generation.saturating_add(1);
        if let Some(reload) = &self.reload {
            reload.apply(*generation, &config);
        }
        Ok(config)
    }
}

impl fmt::Debug for ManagementHandlerOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementHandlerOwner")
            .field("authenticator", &self.authenticator)
            .field("config", &self.config)
            .field("reload", &self.reload.is_some())
            .finish_non_exhaustive()
    }
}

pub trait ManagementAuthClock: Send + Sync {
    fn now_ms(&self) -> i64;
}

#[derive(Debug, Default)]
pub struct SystemManagementAuthClock;

impl ManagementAuthClock for SystemManagementAuthClock {
    fn now_ms(&self) -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .ok()
            .and_then(|duration| i64::try_from(duration.as_millis()).ok())
            .unwrap_or(i64::MAX)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ManagementAuthError {
    InvalidConfiguration,
    RemoteDisabled,
    MissingKey,
    InvalidKey,
    Banned { retry_after_seconds: u64 },
    StateUnavailable,
}

impl ManagementAuthError {
    pub fn status(&self) -> u16 {
        match self {
            Self::MissingKey | Self::InvalidKey => 401,
            Self::StateUnavailable => 500,
            Self::InvalidConfiguration | Self::RemoteDisabled | Self::Banned { .. } => 403,
        }
    }

    pub fn message(&self) -> String {
        match self {
            Self::InvalidConfiguration => "remote management key not set".to_owned(),
            Self::RemoteDisabled => "remote management disabled".to_owned(),
            Self::MissingKey => "missing management key".to_owned(),
            Self::InvalidKey => "invalid management key".to_owned(),
            Self::Banned {
                retry_after_seconds,
            } => format!(
                "IP banned due to too many failed attempts. Try again in {retry_after_seconds}s"
            ),
            Self::StateUnavailable => "management authentication unavailable".to_owned(),
        }
    }
}

impl fmt::Display for ManagementAuthError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message())
    }
}

impl std::error::Error for ManagementAuthError {}

#[derive(Debug, Clone, Copy, Default)]
struct AttemptInfo {
    count: u8,
    blocked_until_ms: i64,
    last_activity_ms: i64,
}

/// Management-key verifier with the upstream five-failure/30-minute IP ban.
///
/// CTOX supplies the plaintext key from its secret store at construction. Only
/// a SHA-256 digest remains in this portable component; no process environment
/// fallback is introduced.
pub struct ManagementAuthenticator {
    key_digest: [u8; 32],
    allow_remote: bool,
    attempts: Mutex<HashMap<String, AttemptInfo>>,
    clock: Arc<dyn ManagementAuthClock>,
}

impl ManagementAuthenticator {
    pub fn new(
        key: &str,
        allow_remote: bool,
        clock: Arc<dyn ManagementAuthClock>,
    ) -> Result<Self, ManagementAuthError> {
        if key.is_empty() {
            return Err(ManagementAuthError::InvalidConfiguration);
        }
        Ok(Self {
            key_digest: Sha256::digest(key.as_bytes()).into(),
            allow_remote,
            attempts: Mutex::new(HashMap::new()),
            clock,
        })
    }

    pub fn authenticate(
        &self,
        client_ip: &str,
        local_client: bool,
        provided: Option<&str>,
    ) -> Result<(), ManagementAuthError> {
        let now_ms = self.clock.now_ms();
        let mut attempts = self
            .attempts
            .lock()
            .map_err(|_| ManagementAuthError::StateUnavailable)?;
        attempts.retain(|_, attempt| {
            attempt.blocked_until_ms > now_ms
                || now_ms.saturating_sub(attempt.last_activity_ms) <= MAX_IDLE_MS
        });
        let attempt = attempts.get_mut(client_ip);
        if let Some(attempt) = attempt {
            if attempt.blocked_until_ms > now_ms {
                let remaining = attempt.blocked_until_ms.saturating_sub(now_ms);
                let seconds = u64::try_from((remaining.saturating_add(999)) / 1_000).unwrap_or(0);
                return Err(ManagementAuthError::Banned {
                    retry_after_seconds: seconds,
                });
            }
            if attempt.blocked_until_ms != 0 {
                attempt.blocked_until_ms = 0;
                attempt.count = 0;
            }
        }

        if !local_client && !self.allow_remote {
            return Err(ManagementAuthError::RemoteDisabled);
        }

        let provided = match provided {
            Some(value) if !value.is_empty() => value,
            _ => {
                record_failure(&mut attempts, client_ip, now_ms);
                return Err(ManagementAuthError::MissingKey);
            }
        };
        let provided_digest: [u8; 32] = Sha256::digest(provided.as_bytes()).into();
        if self.key_digest.ct_eq(&provided_digest).unwrap_u8() != 1 {
            record_failure(&mut attempts, client_ip, now_ms);
            return Err(ManagementAuthError::InvalidKey);
        }
        if let Some(attempt) = attempts.get_mut(client_ip) {
            attempt.count = 0;
            attempt.blocked_until_ms = 0;
            attempt.last_activity_ms = now_ms;
        }
        Ok(())
    }
}

impl fmt::Debug for ManagementAuthenticator {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ManagementAuthenticator")
            .field("key_digest", &"[REDACTED]")
            .field("allow_remote", &self.allow_remote)
            .finish()
    }
}

fn record_failure(attempts: &mut HashMap<String, AttemptInfo>, client_ip: &str, now_ms: i64) {
    let attempt = attempts.entry(client_ip.to_owned()).or_default();
    attempt.count = attempt.count.saturating_add(1);
    attempt.last_activity_ms = now_ms;
    if attempt.count >= MAX_FAILURES {
        attempt.blocked_until_ms = now_ms.saturating_add(BAN_DURATION_MS);
        attempt.count = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI64, Ordering};

    #[derive(Default)]
    struct FixedClock(AtomicI64);

    impl ManagementAuthClock for FixedClock {
        fn now_ms(&self) -> i64 {
            self.0.load(Ordering::SeqCst)
        }
    }

    #[test]
    fn fifth_failure_bans_even_the_correct_local_key_until_expiry() {
        let clock = Arc::new(FixedClock::default());
        let auth = ManagementAuthenticator::new("test-secret", false, clock.clone()).unwrap();
        for _ in 0..5 {
            assert_eq!(
                auth.authenticate("127.0.0.1", true, Some("wrong")),
                Err(ManagementAuthError::InvalidKey)
            );
        }
        assert_eq!(
            auth.authenticate("127.0.0.1", true, Some("test-secret")),
            Err(ManagementAuthError::Banned {
                retry_after_seconds: 1_800
            })
        );
        clock.0.store(BAN_DURATION_MS, Ordering::SeqCst);
        assert_eq!(
            auth.authenticate("127.0.0.1", true, Some("test-secret")),
            Ok(())
        );
    }

    #[test]
    fn remote_policy_and_debug_fail_closed_without_secret_disclosure() {
        assert_eq!(
            ManagementAuthenticator::new("", false, Arc::new(FixedClock::default())).unwrap_err(),
            ManagementAuthError::InvalidConfiguration
        );
        let auth =
            ManagementAuthenticator::new("do-not-render", false, Arc::new(FixedClock::default()))
                .unwrap();
        assert_eq!(
            auth.authenticate("203.0.113.1", false, Some("do-not-render")),
            Err(ManagementAuthError::RemoteDisabled)
        );
        assert!(!format!("{auth:?}").contains("do-not-render"));
    }
}

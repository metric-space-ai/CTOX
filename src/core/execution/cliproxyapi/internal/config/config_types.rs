// ref: internal/config/config_types.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashSet;
use std::fmt;
use std::time::Duration;

use chrono_tz::Tz;
use serde::{Deserialize, Serialize};

use crate::internal::auth::antigravity::{
    AntigravityCredentialHandles, AntigravitySecretHandle, AntigravitySecretKind,
};
use crate::internal::auth::claude::{
    ClaudeCredentialHandles, ClaudeSecretHandle, ClaudeSecretKind,
};
use crate::internal::auth::codex::{CodexCredentialHandles, CodexSecretHandle, CodexSecretKind};
use crate::internal::config::weight::validate_credential_weight;
use crate::internal::runtime::executor::{
    AntigravityTargetError, AntigravityUpstreamTarget, ClaudeDeviceProfile, ClaudeTargetError,
    ClaudeUpstreamTarget, CodexTargetError, CodexUpstreamTarget, DEFAULT_ANTIGRAVITY_BASE_URL,
    DEFAULT_CODEX_BASE_URL,
};
use crate::sdk::cliproxy::auth::{AccountCandidate, SchedulerStrategy};

const DEFAULT_REQUEST_TIMEOUT_MS: u64 = 30_000;
const MAX_REQUEST_TIMEOUT_MS: u64 = 10 * 60 * 1_000;

/// A reference into the host secret store. It is serializable because it never
/// contains the secret value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RuntimeSecretRef {
    pub scope: String,
    pub name: String,
}

impl RuntimeSecretRef {
    fn validate(&self) -> Result<(), RuntimeConfigError> {
        if self.scope.trim().is_empty() || self.name.trim().is_empty() {
            return Err(RuntimeConfigError::InvalidSecretReference);
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaudeDeviceProfileConfig {
    pub user_agent: String,
    pub package_version: String,
    pub runtime_version: String,
    pub os: String,
    pub arch: String,
}

impl ClaudeDeviceProfileConfig {
    pub fn into_profile(self) -> Result<ClaudeDeviceProfile, RuntimeConfigError> {
        ClaudeDeviceProfile::new(
            self.user_agent,
            self.package_version,
            self.runtime_version,
            self.os,
            self.arch,
        )
        .map_err(RuntimeConfigError::Target)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaudeSubscriptionAccountConfig {
    pub id: String,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub priority: i32,
    #[serde(default = "default_account_weight")]
    pub weight: i64,
    #[serde(default)]
    pub websockets: bool,
    #[serde(default)]
    pub models: Vec<String>,
    pub access_token_secret: RuntimeSecretRef,
    pub refresh_token_secret: RuntimeSecretRef,
    #[serde(default = "default_scheme")]
    pub upstream_scheme: String,
    #[serde(default = "default_claude_authority")]
    pub upstream_authority: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url_secret: Option<RuntimeSecretRef>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub device_profile: Option<ClaudeDeviceProfileConfig>,
    /// IANA timezone used for Claude Code's request-local currentDate block.
    /// Empty means UTC; ambient process timezone is deliberately ignored.
    #[serde(default)]
    pub timezone: String,
}

impl ClaudeSubscriptionAccountConfig {
    pub fn credential_handles(&self) -> Result<ClaudeCredentialHandles, RuntimeConfigError> {
        let access = ClaudeSecretHandle::new(
            self.access_token_secret.scope.clone(),
            self.access_token_secret.name.clone(),
            ClaudeSecretKind::AccessToken,
        )
        .map_err(|_| RuntimeConfigError::InvalidSecretReference)?;
        let refresh = ClaudeSecretHandle::new(
            self.refresh_token_secret.scope.clone(),
            self.refresh_token_secret.name.clone(),
            ClaudeSecretKind::RefreshToken,
        )
        .map_err(|_| RuntimeConfigError::InvalidSecretReference)?;
        ClaudeCredentialHandles::new(access, refresh)
            .map_err(|_| RuntimeConfigError::InvalidSecretReference)
    }

    pub fn upstream_target(&self) -> Result<ClaudeUpstreamTarget, RuntimeConfigError> {
        ClaudeUpstreamTarget::new(
            self.upstream_scheme.clone(),
            self.upstream_authority.clone(),
        )
        .map_err(RuntimeConfigError::Target)
    }

    pub fn candidate(&self) -> AccountCandidate {
        AccountCandidate {
            auth_id: self.id.trim().to_owned(),
            provider: "claude".to_owned(),
            priority: self.priority,
            weight: self.weight,
            websocket_enabled: self.websockets,
            supported_models: self.models.clone(),
            disabled: self.disabled,
        }
    }

    pub fn timezone(&self) -> Result<Tz, RuntimeConfigError> {
        let timezone = self.timezone.trim();
        if timezone.is_empty() {
            return Ok(chrono_tz::UTC);
        }
        timezone
            .parse::<Tz>()
            .map_err(|_| RuntimeConfigError::InvalidTimezone)
    }

    fn validate(&self) -> Result<(), RuntimeConfigError> {
        if self.id.trim().is_empty() {
            return Err(RuntimeConfigError::InvalidAccountId);
        }
        validate_credential_weight(Some(self.weight))
            .map_err(|_| RuntimeConfigError::InvalidCredentialWeight)?;
        self.access_token_secret.validate()?;
        self.refresh_token_secret.validate()?;
        if self.access_token_secret == self.refresh_token_secret {
            return Err(RuntimeConfigError::DuplicateSecretReference);
        }
        if let Some(proxy) = &self.proxy_url_secret {
            proxy.validate()?;
        }
        self.credential_handles()?;
        self.upstream_target()?;
        if let Some(profile) = self.device_profile.clone() {
            profile.into_profile()?;
        }
        self.timezone()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CodexSubscriptionAccountConfig {
    pub id: String,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub priority: i32,
    #[serde(default = "default_account_weight")]
    pub weight: i64,
    #[serde(default)]
    pub websockets: bool,
    #[serde(default)]
    pub models: Vec<String>,
    pub id_token_secret: RuntimeSecretRef,
    pub access_token_secret: RuntimeSecretRef,
    pub refresh_token_secret: RuntimeSecretRef,
    #[serde(default = "default_codex_base_url")]
    pub upstream_base_url: String,
    #[serde(default)]
    pub plan_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url_secret: Option<RuntimeSecretRef>,
}

impl CodexSubscriptionAccountConfig {
    pub fn credential_handles(&self) -> Result<CodexCredentialHandles, RuntimeConfigError> {
        let id = CodexSecretHandle::new(
            self.id_token_secret.scope.clone(),
            self.id_token_secret.name.clone(),
            CodexSecretKind::IdToken,
        )
        .map_err(|_| RuntimeConfigError::InvalidSecretReference)?;
        let access = CodexSecretHandle::new(
            self.access_token_secret.scope.clone(),
            self.access_token_secret.name.clone(),
            CodexSecretKind::AccessToken,
        )
        .map_err(|_| RuntimeConfigError::InvalidSecretReference)?;
        let refresh = CodexSecretHandle::new(
            self.refresh_token_secret.scope.clone(),
            self.refresh_token_secret.name.clone(),
            CodexSecretKind::RefreshToken,
        )
        .map_err(|_| RuntimeConfigError::InvalidSecretReference)?;
        CodexCredentialHandles::new(id, access, refresh)
            .map_err(|_| RuntimeConfigError::InvalidSecretReference)
    }

    pub fn candidate(&self) -> AccountCandidate {
        AccountCandidate {
            auth_id: self.id.trim().to_owned(),
            provider: "codex".to_owned(),
            priority: self.priority,
            weight: self.weight,
            websocket_enabled: self.websockets,
            supported_models: self.models.clone(),
            disabled: self.disabled,
        }
    }

    pub fn upstream_target(&self) -> Result<CodexUpstreamTarget, RuntimeConfigError> {
        CodexUpstreamTarget::new(self.upstream_base_url.clone())
            .map_err(RuntimeConfigError::CodexTarget)
    }

    fn validate(&self) -> Result<(), RuntimeConfigError> {
        if self.id.trim().is_empty() {
            return Err(RuntimeConfigError::InvalidAccountId);
        }
        validate_credential_weight(Some(self.weight))
            .map_err(|_| RuntimeConfigError::InvalidCredentialWeight)?;
        self.id_token_secret.validate()?;
        self.access_token_secret.validate()?;
        self.refresh_token_secret.validate()?;
        let mut refs = HashSet::new();
        for secret_ref in [
            &self.id_token_secret,
            &self.access_token_secret,
            &self.refresh_token_secret,
        ] {
            if !refs.insert((&secret_ref.scope, &secret_ref.name)) {
                return Err(RuntimeConfigError::DuplicateSecretReference);
            }
        }
        if let Some(proxy) = &self.proxy_url_secret {
            proxy.validate()?;
        }
        if self.plan_type.len() > 64 || self.plan_type.chars().any(char::is_control) {
            return Err(RuntimeConfigError::InvalidPlanType);
        }
        self.credential_handles()?;
        self.upstream_target()?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AntigravitySubscriptionAccountConfig {
    pub id: String,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub priority: i32,
    #[serde(default = "default_account_weight")]
    pub weight: i64,
    #[serde(default)]
    pub websockets: bool,
    #[serde(default)]
    pub models: Vec<String>,
    pub access_token_secret: RuntimeSecretRef,
    pub refresh_token_secret: RuntimeSecretRef,
    pub state_secret: RuntimeSecretRef,
    #[serde(default = "default_antigravity_base_url")]
    pub upstream_base_url: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub proxy_url_secret: Option<RuntimeSecretRef>,
}

impl AntigravitySubscriptionAccountConfig {
    pub fn credential_handles(&self) -> Result<AntigravityCredentialHandles, RuntimeConfigError> {
        let handle = |secret: &RuntimeSecretRef, kind| {
            AntigravitySecretHandle::new(secret.scope.clone(), secret.name.clone(), kind)
                .map_err(|_| RuntimeConfigError::InvalidSecretReference)
        };
        AntigravityCredentialHandles::new(
            handle(
                &self.access_token_secret,
                AntigravitySecretKind::AccessToken,
            )?,
            handle(
                &self.refresh_token_secret,
                AntigravitySecretKind::RefreshToken,
            )?,
            handle(&self.state_secret, AntigravitySecretKind::State)?,
        )
        .map_err(|_| RuntimeConfigError::InvalidSecretReference)
    }

    pub fn candidate(&self) -> AccountCandidate {
        AccountCandidate {
            auth_id: self.id.trim().to_owned(),
            provider: "antigravity".to_owned(),
            priority: self.priority,
            weight: self.weight,
            websocket_enabled: self.websockets,
            supported_models: self.models.clone(),
            disabled: self.disabled,
        }
    }

    pub fn upstream_target(&self) -> Result<AntigravityUpstreamTarget, RuntimeConfigError> {
        AntigravityUpstreamTarget::new(self.upstream_base_url.clone())
            .map_err(RuntimeConfigError::AntigravityTarget)
    }

    fn validate(&self) -> Result<(), RuntimeConfigError> {
        if self.id.trim().is_empty() {
            return Err(RuntimeConfigError::InvalidAccountId);
        }
        validate_credential_weight(Some(self.weight))
            .map_err(|_| RuntimeConfigError::InvalidCredentialWeight)?;
        let mut refs = HashSet::new();
        for secret_ref in [
            &self.access_token_secret,
            &self.refresh_token_secret,
            &self.state_secret,
        ] {
            secret_ref.validate()?;
            if !refs.insert((&secret_ref.scope, &secret_ref.name)) {
                return Err(RuntimeConfigError::DuplicateSecretReference);
            }
        }
        if let Some(proxy) = &self.proxy_url_secret {
            proxy.validate()?;
        }
        self.credential_handles()?;
        self.upstream_target()?;
        Ok(())
    }
}

fn default_codex_base_url() -> String {
    DEFAULT_CODEX_BASE_URL.to_owned()
}

fn default_antigravity_base_url() -> String {
    DEFAULT_ANTIGRAVITY_BASE_URL.to_owned()
}

fn default_scheme() -> String {
    "https".to_owned()
}

fn default_claude_authority() -> String {
    "api.anthropic.com".to_owned()
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CliproxyRuntimeConfig {
    #[serde(default = "default_request_timeout_ms")]
    pub request_timeout_ms: u64,
    #[serde(default)]
    pub routing_strategy: SchedulerStrategy,
    #[serde(default)]
    pub claude_accounts: Vec<ClaudeSubscriptionAccountConfig>,
    #[serde(default)]
    pub codex_accounts: Vec<CodexSubscriptionAccountConfig>,
    #[serde(default)]
    pub antigravity_accounts: Vec<AntigravitySubscriptionAccountConfig>,
}

fn default_request_timeout_ms() -> u64 {
    DEFAULT_REQUEST_TIMEOUT_MS
}

fn default_account_weight() -> i64 {
    1
}

impl CliproxyRuntimeConfig {
    pub fn validate(self) -> Result<ValidatedRuntimeConfig, RuntimeConfigError> {
        if self.request_timeout_ms == 0 || self.request_timeout_ms > MAX_REQUEST_TIMEOUT_MS {
            return Err(RuntimeConfigError::InvalidTimeout);
        }
        let mut ids = HashSet::new();
        for account in &self.claude_accounts {
            account.validate()?;
            if !ids.insert(account.id.trim().to_owned()) {
                return Err(RuntimeConfigError::DuplicateAccountId);
            }
        }
        for account in &self.codex_accounts {
            account.validate()?;
            if !ids.insert(account.id.trim().to_owned()) {
                return Err(RuntimeConfigError::DuplicateAccountId);
            }
        }
        for account in &self.antigravity_accounts {
            account.validate()?;
            if !ids.insert(account.id.trim().to_owned()) {
                return Err(RuntimeConfigError::DuplicateAccountId);
            }
        }
        if !self.claude_accounts.iter().any(|account| !account.disabled)
            && !self.codex_accounts.iter().any(|account| !account.disabled)
            && !self
                .antigravity_accounts
                .iter()
                .any(|account| !account.disabled)
        {
            return Err(RuntimeConfigError::NoEnabledAccounts);
        }
        Ok(ValidatedRuntimeConfig(self))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ValidatedRuntimeConfig(CliproxyRuntimeConfig);

impl ValidatedRuntimeConfig {
    pub fn as_config(&self) -> &CliproxyRuntimeConfig {
        &self.0
    }

    pub fn into_config(self) -> CliproxyRuntimeConfig {
        self.0
    }

    pub fn request_timeout(&self) -> Duration {
        Duration::from_millis(self.0.request_timeout_ms)
    }

    pub fn routing_strategy(&self) -> SchedulerStrategy {
        self.0.routing_strategy
    }

    pub fn claude_accounts(&self) -> &[ClaudeSubscriptionAccountConfig] {
        &self.0.claude_accounts
    }

    pub fn claude_candidates(&self) -> Vec<AccountCandidate> {
        self.0
            .claude_accounts
            .iter()
            .map(ClaudeSubscriptionAccountConfig::candidate)
            .collect()
    }

    pub fn codex_accounts(&self) -> &[CodexSubscriptionAccountConfig] {
        &self.0.codex_accounts
    }

    pub fn codex_candidates(&self) -> Vec<AccountCandidate> {
        self.0
            .codex_accounts
            .iter()
            .map(CodexSubscriptionAccountConfig::candidate)
            .collect()
    }

    pub fn antigravity_accounts(&self) -> &[AntigravitySubscriptionAccountConfig] {
        &self.0.antigravity_accounts
    }

    pub fn antigravity_candidates(&self) -> Vec<AccountCandidate> {
        self.0
            .antigravity_accounts
            .iter()
            .map(AntigravitySubscriptionAccountConfig::candidate)
            .collect()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeConfigError {
    InvalidTimeout,
    InvalidAccountId,
    DuplicateAccountId,
    NoEnabledAccounts,
    InvalidSecretReference,
    DuplicateSecretReference,
    InvalidPlanType,
    InvalidCredentialWeight,
    InvalidTimezone,
    Target(ClaudeTargetError),
    CodexTarget(CodexTargetError),
    AntigravityTarget(AntigravityTargetError),
}

impl fmt::Display for RuntimeConfigError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidTimeout => "proxy request timeout is invalid",
            Self::InvalidAccountId => "proxy account id is invalid",
            Self::DuplicateAccountId => "proxy account id is duplicated",
            Self::NoEnabledAccounts => "proxy has no enabled accounts",
            Self::InvalidSecretReference => "proxy secret reference is invalid",
            Self::DuplicateSecretReference => "proxy credential references must be distinct",
            Self::InvalidPlanType => "proxy subscription plan type is invalid",
            Self::InvalidCredentialWeight => "proxy credential weight is invalid",
            Self::InvalidTimezone => "proxy credential timezone is invalid",
            Self::Target(_) | Self::CodexTarget(_) | Self::AntigravityTarget(_) => {
                "proxy upstream target is invalid"
            }
        })
    }
}

impl std::error::Error for RuntimeConfigError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn account(id: &str) -> ClaudeSubscriptionAccountConfig {
        ClaudeSubscriptionAccountConfig {
            id: id.to_owned(),
            disabled: false,
            priority: 3,
            weight: 1,
            websockets: false,
            models: Vec::new(),
            access_token_secret: RuntimeSecretRef {
                scope: "subscriptions".to_owned(),
                name: format!("{id}-access"),
            },
            refresh_token_secret: RuntimeSecretRef {
                scope: "subscriptions".to_owned(),
                name: format!("{id}-refresh"),
            },
            upstream_scheme: "https".to_owned(),
            upstream_authority: "api.anthropic.com".to_owned(),
            proxy_url_secret: None,
            device_profile: None,
            timezone: String::new(),
        }
    }

    #[test]
    fn claude_timezone_is_typed_and_invalid_names_fail_validation() {
        let mut valid = account("account-timezone");
        valid.timezone = " Pacific/Honolulu ".to_owned();
        assert_eq!(valid.timezone().unwrap(), chrono_tz::Pacific::Honolulu);

        valid.timezone = "Mars/Olympus_Mons".to_owned();
        assert_eq!(valid.validate(), Err(RuntimeConfigError::InvalidTimezone));
    }

    fn codex_account(id: &str) -> CodexSubscriptionAccountConfig {
        let secret = |suffix: &str| RuntimeSecretRef {
            scope: "subscriptions".to_owned(),
            name: format!("{id}-{suffix}"),
        };
        CodexSubscriptionAccountConfig {
            id: id.to_owned(),
            disabled: false,
            priority: 4,
            weight: 1,
            websockets: false,
            models: Vec::new(),
            id_token_secret: secret("id"),
            access_token_secret: secret("access"),
            refresh_token_secret: secret("refresh"),
            upstream_base_url: DEFAULT_CODEX_BASE_URL.to_owned(),
            plan_type: "pro".to_owned(),
            proxy_url_secret: None,
        }
    }

    fn antigravity_account(id: &str) -> AntigravitySubscriptionAccountConfig {
        let secret = |suffix: &str| RuntimeSecretRef {
            scope: "subscriptions".to_owned(),
            name: format!("{id}-{suffix}"),
        };
        AntigravitySubscriptionAccountConfig {
            id: id.to_owned(),
            disabled: false,
            priority: 5,
            weight: 1,
            websockets: false,
            models: Vec::new(),
            access_token_secret: secret("access"),
            refresh_token_secret: secret("refresh"),
            state_secret: secret("state"),
            upstream_base_url: DEFAULT_ANTIGRAVITY_BASE_URL.to_owned(),
            proxy_url_secret: None,
        }
    }

    #[test]
    fn valid_config_builds_handles_targets_candidates_and_timeout() {
        let config = CliproxyRuntimeConfig {
            request_timeout_ms: 45_000,
            routing_strategy: SchedulerStrategy::RoundRobin,
            claude_accounts: vec![account("account-a")],
            codex_accounts: vec![codex_account("codex-a")],
            antigravity_accounts: vec![antigravity_account("antigravity-a")],
        }
        .validate()
        .unwrap();
        assert_eq!(config.request_timeout(), Duration::from_secs(45));
        assert_eq!(config.claude_candidates()[0].priority, 3);
        assert_eq!(config.codex_candidates()[0].provider, "codex");
        assert_eq!(config.antigravity_candidates()[0].provider, "antigravity");
        assert!(config.antigravity_accounts()[0]
            .credential_handles()
            .is_ok());
        assert!(config.codex_accounts()[0].credential_handles().is_ok());
        assert!(config.codex_accounts()[0].upstream_target().is_ok());
        assert!(config.claude_accounts()[0].credential_handles().is_ok());
        assert!(config.claude_accounts()[0]
            .upstream_target()
            .unwrap()
            .is_anthropic_api());
    }

    #[test]
    fn serialized_config_contains_handles_but_no_credentials() {
        let encoded = serde_json::to_string(&CliproxyRuntimeConfig {
            request_timeout_ms: 30_000,
            routing_strategy: SchedulerStrategy::RoundRobin,
            claude_accounts: vec![account("account-a")],
            codex_accounts: vec![codex_account("codex-a")],
            antigravity_accounts: vec![antigravity_account("antigravity-a")],
        })
        .unwrap();
        assert!(encoded.contains("account-a-access"));
        assert!(encoded.contains("codex-a-id"));
        assert!(encoded.contains("antigravity-a-state"));
        assert!(!encoded.contains("Bearer "));
        assert!(!encoded.contains("refresh_token\":"));
    }

    #[test]
    fn duplicate_accounts_and_secret_handles_fail_closed() {
        assert_eq!(
            CliproxyRuntimeConfig {
                request_timeout_ms: 30_000,
                routing_strategy: SchedulerStrategy::RoundRobin,
                claude_accounts: vec![account("same"), account("same")],
                codex_accounts: Vec::new(),
                antigravity_accounts: Vec::new(),
            }
            .validate(),
            Err(RuntimeConfigError::DuplicateAccountId)
        );
        let mut invalid = account("account-a");
        invalid.refresh_token_secret = invalid.access_token_secret.clone();
        assert_eq!(
            CliproxyRuntimeConfig {
                request_timeout_ms: 30_000,
                routing_strategy: SchedulerStrategy::RoundRobin,
                claude_accounts: vec![invalid],
                codex_accounts: Vec::new(),
                antigravity_accounts: Vec::new(),
            }
            .validate(),
            Err(RuntimeConfigError::DuplicateSecretReference)
        );
    }

    #[test]
    fn unknown_fields_and_header_injection_are_rejected() {
        let unknown = r#"{"request_timeout_ms":30000,"claude_accounts":[],"token":"secret"}"#;
        assert!(serde_json::from_str::<CliproxyRuntimeConfig>(unknown).is_err());
        let mut invalid = account("account-a");
        invalid.device_profile = Some(ClaudeDeviceProfileConfig {
            user_agent: "claude-cli/2.1.63\r\nX-Evil: yes".to_owned(),
            package_version: "0.74.0".to_owned(),
            runtime_version: "v24.3.0".to_owned(),
            os: "MacOS".to_owned(),
            arch: "arm64".to_owned(),
        });
        assert_eq!(
            CliproxyRuntimeConfig {
                request_timeout_ms: 30_000,
                routing_strategy: SchedulerStrategy::RoundRobin,
                claude_accounts: vec![invalid],
                codex_accounts: Vec::new(),
                antigravity_accounts: Vec::new(),
            }
            .validate(),
            Err(RuntimeConfigError::Target(
                ClaudeTargetError::InvalidFingerprint
            ))
        );
    }
}

// Origin: CTOX
// License: AGPL-3.0-only

//! Typed, account-scoped Kimi Coding Plan configuration.
//!
//! This module owns only durable account topology and opaque secret handles.
//! It is deliberately independent from CTOX's active main model and never
//! persists API-key material in the account payload.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::path::Path;

const STORAGE_KEY: &str = "kimi_coding_accounts_v1";
pub const CONFIG_SCHEMA: &str = "ctox.kimi-coding-accounts.v1";
pub const DEFAULT_MODEL: &str = "k3[1m]";
pub const SUPPORTED_MODELS: &[&str] = &[DEFAULT_MODEL];
pub const ANTHROPIC_BASE_URL: &str = "https://api.kimi.com/coding/";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum KimiCodingEndpointProfile {
    #[default]
    KimiCoding,
}

impl KimiCodingEndpointProfile {
    pub fn base_url(self) -> &'static str {
        match self {
            Self::KimiCoding => ANTHROPIC_BASE_URL,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KimiSecretRef {
    pub scope: String,
    pub name: String,
}

impl KimiSecretRef {
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.scope.trim().is_empty()
                && !self.name.trim().is_empty()
                && !self.scope.chars().any(char::is_control)
                && !self.name.chars().any(char::is_control),
            "Kimi secret reference is invalid"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KimiCodingAccount {
    /// Stable account identity. Updating an account with the same id replaces
    /// its topology in place; callers must never infer identity from a model.
    pub id: String,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub models: Vec<String>,
    pub api_key_secret: KimiSecretRef,
    #[serde(default)]
    pub endpoint_profile: KimiCodingEndpointProfile,
}

impl KimiCodingAccount {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.id.is_empty()
                && self
                    .id
                    .chars()
                    .all(|value| value.is_ascii_alphanumeric() || matches!(value, '-' | '_' | '.')),
            "Kimi account id must contain only ASCII letters, digits, '.', '_' or '-'"
        );
        self.api_key_secret.validate()?;
        anyhow::ensure!(
            self.models.iter().all(|model| {
                let model = model.trim();
                SUPPORTED_MODELS.contains(&model)
            }),
            "Kimi coding account contains an unsupported model"
        );
        Ok(())
    }

    pub fn effective_models(&self) -> Vec<String> {
        if self.models.is_empty() {
            vec![DEFAULT_MODEL.to_owned()]
        } else {
            self.models
                .iter()
                .map(|model| model.trim().to_owned())
                .collect()
        }
    }

    pub fn credential_is_ready(&self, root: &Path) -> anyhow::Result<bool> {
        crate::secrets::secret_exists(
            root,
            self.api_key_secret.scope.trim(),
            self.api_key_secret.name.trim(),
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KimiCodingAccountsConfig {
    #[serde(default = "config_schema")]
    pub schema: String,
    #[serde(default)]
    pub accounts: Vec<KimiCodingAccount>,
}

impl Default for KimiCodingAccountsConfig {
    fn default() -> Self {
        Self {
            schema: CONFIG_SCHEMA.to_owned(),
            accounts: Vec::new(),
        }
    }
}

impl KimiCodingAccountsConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == CONFIG_SCHEMA,
            "Kimi config schema is invalid"
        );
        let mut ids = HashSet::new();
        let mut secrets = HashSet::new();
        for account in &self.accounts {
            account.validate()?;
            anyhow::ensure!(ids.insert(account.id.as_str()), "duplicate Kimi account id");
            anyhow::ensure!(
                secrets.insert((
                    account.api_key_secret.scope.trim(),
                    account.api_key_secret.name.trim()
                )),
                "Kimi accounts must not alias one secret reference"
            );
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum KimiCodingAccountPhase {
    Disabled,
    MissingCredential,
    Ready,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct KimiCodingAccountStatus {
    pub id: String,
    pub phase: KimiCodingAccountPhase,
    pub models: Vec<String>,
    pub endpoint_profile: KimiCodingEndpointProfile,
}

fn config_schema() -> String {
    CONFIG_SCHEMA.to_owned()
}

pub fn load_accounts(root: &Path) -> anyhow::Result<KimiCodingAccountsConfig> {
    let config: KimiCodingAccountsConfig =
        crate::persistence::load_json_payload(root, STORAGE_KEY)?.unwrap_or_default();
    config.validate()?;
    Ok(config)
}

pub fn store_accounts(root: &Path, config: &KimiCodingAccountsConfig) -> anyhow::Result<()> {
    config.validate()?;
    crate::persistence::store_json_payload(root, STORAGE_KEY, Some(config))
}

/// Returns a non-secret projection and performs only encrypted-store metadata
/// checks. API-key material is never resolved for status rendering.
pub fn account_statuses(root: &Path) -> anyhow::Result<Vec<KimiCodingAccountStatus>> {
    load_accounts(root)?
        .accounts
        .into_iter()
        .map(|account| {
            let models = account.effective_models();
            let phase = if account.disabled {
                KimiCodingAccountPhase::Disabled
            } else if account.credential_is_ready(root)? {
                KimiCodingAccountPhase::Ready
            } else {
                KimiCodingAccountPhase::MissingCredential
            };
            Ok(KimiCodingAccountStatus {
                id: account.id,
                phase,
                models,
                endpoint_profile: account.endpoint_profile,
            })
        })
        .collect()
}

/// Adds or replaces exactly one stable account identity without touching its
/// encrypted credential record.
pub fn upsert_account(root: &Path, account: KimiCodingAccount) -> anyhow::Result<()> {
    let _lifecycle = crate::secrets::credential_lifecycle_guard();
    account.validate()?;
    let mut config = load_accounts(root)?;
    config.accounts.retain(|current| current.id != account.id);
    config.accounts.push(account);
    config
        .accounts
        .sort_by(|left, right| left.id.cmp(&right.id));
    store_accounts(root, &config)
}

/// Rotates only the selected account's secret record. The topology and secret
/// handle stay stable across rotations.
pub fn rotate_api_key(root: &Path, account_id: &str, api_key: &str) -> anyhow::Result<()> {
    let _lifecycle = crate::secrets::credential_lifecycle_guard();
    let api_key = api_key.trim();
    anyhow::ensure!(!api_key.is_empty(), "Kimi API key is required");
    let config = load_accounts(root)?;
    let account = config
        .accounts
        .iter()
        .find(|account| account.id == account_id.trim())
        .with_context(|| format!("Kimi coding account does not exist: {account_id}"))?;
    crate::secrets::write_secret_record(
        root,
        account.api_key_secret.scope.trim(),
        account.api_key_secret.name.trim(),
        api_key,
        Some("Kimi Coding Plan API key".to_owned()),
        serde_json::json!({
            "provider": "kimi",
            "access_mode": "coding_plan",
            "account_id": account.id,
        }),
    )?;
    Ok(())
}

/// Disconnects fail-closed: durable topology is removed first, then the
/// encrypted secret is cleaned up. Cleanup failure can therefore leave only
/// an unreferenced orphan, never an advertised account with broken state.
pub fn disconnect_account(root: &Path, account_id: &str) -> anyhow::Result<bool> {
    let _lifecycle = crate::secrets::credential_lifecycle_guard();
    disconnect_account_with(root, account_id, |account| {
        if account.credential_is_ready(root)? {
            crate::secrets::delete_secret_record(
                root,
                account.api_key_secret.scope.trim(),
                account.api_key_secret.name.trim(),
            )?;
        }
        Ok(())
    })
}

fn disconnect_account_with(
    root: &Path,
    account_id: &str,
    delete_secret: impl FnOnce(&KimiCodingAccount) -> anyhow::Result<()>,
) -> anyhow::Result<bool> {
    let account_id = account_id.trim();
    let mut config = load_accounts(root)?;
    let Some(index) = config
        .accounts
        .iter()
        .position(|account| account.id == account_id)
    else {
        return Ok(false);
    };
    let account = config.accounts[index].clone();
    config.accounts.remove(index);
    store_accounts(root, &config)?;
    delete_secret(&account)?;
    Ok(true)
}

pub fn ready_accounts(root: &Path) -> anyhow::Result<Vec<KimiCodingAccount>> {
    load_accounts(root)?
        .accounts
        .into_iter()
        .filter(|account| !account.disabled)
        .filter_map(|account| match account.credential_is_ready(root) {
            Ok(true) => Some(Ok(account)),
            Ok(false) => None,
            Err(error) => Some(Err(error)),
        })
        .collect()
}

pub fn resolve_ready_account(root: &Path, account_id: &str) -> anyhow::Result<KimiCodingAccount> {
    let account_id = account_id.trim();
    anyhow::ensure!(!account_id.is_empty(), "Kimi account id is required");
    let mut matches = ready_accounts(root)?
        .into_iter()
        .filter(|account| account.id == account_id);
    let account = matches
        .next()
        .with_context(|| format!("Kimi coding account is unavailable: {account_id}"))?;
    anyhow::ensure!(matches.next().is_none(), "Kimi coding account is ambiguous");
    Ok(account)
}

/// Resolves API-key material only for an already selected account and bounded
/// native use. Callers must not serialize the returned value.
pub fn read_api_key(root: &Path, account: &KimiCodingAccount) -> anyhow::Result<String> {
    crate::secrets::read_secret_value(
        root,
        account.api_key_secret.scope.trim(),
        account.api_key_secret.name.trim(),
    )
    .context("Kimi coding credential is unavailable")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn account(id: &str) -> KimiCodingAccount {
        KimiCodingAccount {
            id: id.to_owned(),
            disabled: false,
            models: Vec::new(),
            api_key_secret: KimiSecretRef {
                scope: "provider-subscriptions".to_owned(),
                name: format!("{id}-api-key"),
            },
            endpoint_profile: KimiCodingEndpointProfile::KimiCoding,
        }
    }

    #[test]
    fn account_config_round_trips_without_secret_material() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let config = KimiCodingAccountsConfig {
            schema: CONFIG_SCHEMA.to_owned(),
            accounts: vec![account("kimi-primary")],
        };
        store_accounts(temp.path(), &config)?;
        assert_eq!(load_accounts(temp.path())?, config);
        assert_eq!(config.accounts[0].effective_models(), [DEFAULT_MODEL]);
        let encoded = serde_json::to_string(&config)?;
        assert!(!encoded.contains("secret-value"));
        assert_eq!(
            config.accounts[0].endpoint_profile.base_url(),
            ANTHROPIC_BASE_URL
        );

        let mut unsupported = account("unsupported");
        unsupported.models = vec!["moonshot-v1".to_owned()];
        assert!(unsupported.validate().is_err());
        Ok(())
    }

    #[test]
    fn readiness_requires_the_independent_secret_reference() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        let configured = account("kimi-one");
        store_accounts(
            root,
            &KimiCodingAccountsConfig {
                accounts: vec![configured.clone()],
                ..KimiCodingAccountsConfig::default()
            },
        )?;
        assert!(ready_accounts(root)?.is_empty());
        assert_eq!(
            account_statuses(root)?[0].phase,
            KimiCodingAccountPhase::MissingCredential
        );

        rotate_api_key(root, "kimi-one", "secret-value")?;
        assert_eq!(resolve_ready_account(root, "kimi-one")?.id, "kimi-one");
        assert_eq!(read_api_key(root, &configured)?, "secret-value");
        rotate_api_key(root, "kimi-one", "rotated-value")?;
        assert_eq!(read_api_key(root, &configured)?, "rotated-value");

        let connection = rusqlite::Connection::open(crate::secrets::secret_store_path(root))?;
        connection.execute_batch(
            r#"
            CREATE TRIGGER reject_kimi_coding_rotation
            BEFORE UPDATE ON ctox_secret_records
            WHEN NEW.secret_name = 'kimi-one-api-key'
            BEGIN
                SELECT RAISE(ABORT, 'forced Kimi Coding rollback');
            END;
            "#,
        )?;
        drop(connection);
        assert!(rotate_api_key(root, "kimi-one", "rejected-value").is_err());
        assert_eq!(read_api_key(root, &configured)?, "rotated-value");
        assert_eq!(
            account_statuses(root)?[0].phase,
            KimiCodingAccountPhase::Ready
        );

        let mut disabled = configured;
        disabled.disabled = true;
        upsert_account(root, disabled)?;
        assert!(ready_accounts(root)?.is_empty());
        assert_eq!(
            account_statuses(root)?[0].phase,
            KimiCodingAccountPhase::Disabled
        );
        assert!(disconnect_account(root, "kimi-one")?);
        assert!(!disconnect_account(root, "kimi-one")?);
        assert!(!crate::secrets::secret_exists(
            root,
            "provider-subscriptions",
            "kimi-one-api-key"
        )?);
        Ok(())
    }

    #[test]
    fn duplicate_ids_and_secret_references_fail_closed() {
        let first = account("kimi-one");
        let mut duplicate_id = account("kimi-one");
        duplicate_id.api_key_secret.name = "another-key".to_owned();
        assert!(KimiCodingAccountsConfig {
            accounts: vec![first.clone(), duplicate_id],
            ..KimiCodingAccountsConfig::default()
        }
        .validate()
        .is_err());

        let mut duplicate_secret = account("kimi-two");
        duplicate_secret.api_key_secret = first.api_key_secret.clone();
        assert!(KimiCodingAccountsConfig {
            accounts: vec![first, duplicate_secret],
            ..KimiCodingAccountsConfig::default()
        }
        .validate()
        .is_err());
    }

    #[test]
    fn disconnect_cleanup_failure_leaves_only_an_unreferenced_secret() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        upsert_account(root, account("kimi-orphan"))?;
        rotate_api_key(root, "kimi-orphan", "secret")?;

        let error = disconnect_account_with(root, "kimi-orphan", |_| {
            anyhow::bail!("synthetic cleanup failure")
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("synthetic cleanup failure"));
        assert!(load_accounts(root)?.accounts.is_empty());
        assert!(crate::secrets::secret_exists(
            root,
            "provider-subscriptions",
            "kimi-orphan-api-key"
        )?);
        assert!(ready_accounts(root)?.is_empty());
        Ok(())
    }
}

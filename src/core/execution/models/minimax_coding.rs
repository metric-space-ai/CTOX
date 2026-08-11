// Origin: CTOX
// License: AGPL-3.0-only

//! Typed, account-scoped MiniMax Coding Plan configuration.
//!
//! The coding account is deliberately independent from CTOX's active main
//! model and from the `ctox_proxy` provider. Only an opaque account id, fixed
//! MiniMax endpoint profile and encrypted-store secret handle are persisted;
//! the API key is resolved by the native owner only for a bounded turn or an
//! explicit capacity refresh.

use anyhow::Context;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashSet;
use std::io::Read;
use std::path::Path;
use std::time::Duration;

const STORAGE_KEY: &str = "minimax_coding_accounts_v1";
pub const CONFIG_SCHEMA: &str = "ctox.minimax-coding-accounts.v1";
pub const DEFAULT_MODEL: &str = "MiniMax-M3";
pub const ANTHROPIC_BASE_URL: &str = "https://api.minimax.io/anthropic";
pub const CAPACITY_URL: &str = "https://www.minimax.io/v1/token_plan/remains";

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxCodingEndpointProfile {
    #[default]
    GlobalAnthropic,
}

impl MiniMaxCodingEndpointProfile {
    pub fn base_url(self) -> &'static str {
        match self {
            Self::GlobalAnthropic => ANTHROPIC_BASE_URL,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxSecretRef {
    pub scope: String,
    pub name: String,
}

impl MiniMaxSecretRef {
    fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.scope.trim().is_empty()
                && !self.name.trim().is_empty()
                && !self.scope.chars().any(char::is_control)
                && !self.name.chars().any(char::is_control),
            "MiniMax secret reference is invalid"
        );
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MiniMaxCodingAccount {
    pub id: String,
    #[serde(default)]
    pub disabled: bool,
    #[serde(default)]
    pub models: Vec<String>,
    pub api_key_secret: MiniMaxSecretRef,
    #[serde(default)]
    pub endpoint_profile: MiniMaxCodingEndpointProfile,
}

impl MiniMaxCodingAccount {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            !self.id.is_empty()
                && self
                    .id
                    .chars()
                    .all(|value| value.is_ascii_alphanumeric() || matches!(value, '-' | '_' | '.')),
            "MiniMax account id must contain only ASCII letters, digits, '.', '_' or '-'"
        );
        self.api_key_secret.validate()?;
        anyhow::ensure!(
            self.models.iter().all(|model| {
                let model = model.trim();
                !model.is_empty()
                    && !model.chars().any(char::is_control)
                    && model.to_ascii_lowercase().starts_with("minimax-")
            }),
            "MiniMax coding account contains an invalid model"
        );
        let mut models = HashSet::new();
        anyhow::ensure!(
            self.models
                .iter()
                .all(|model| models.insert(model.trim().to_ascii_lowercase())),
            "MiniMax coding account contains a duplicate model"
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
pub struct MiniMaxCodingAccountsConfig {
    #[serde(default = "config_schema")]
    pub schema: String,
    #[serde(default)]
    pub accounts: Vec<MiniMaxCodingAccount>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MiniMaxCodingAccountPhase {
    Disabled,
    MissingCredential,
    Ready,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct MiniMaxCodingAccountStatus {
    pub id: String,
    pub phase: MiniMaxCodingAccountPhase,
    pub models: Vec<String>,
    pub endpoint_profile: MiniMaxCodingEndpointProfile,
}

impl Default for MiniMaxCodingAccountsConfig {
    fn default() -> Self {
        Self {
            schema: CONFIG_SCHEMA.to_owned(),
            accounts: Vec::new(),
        }
    }
}

impl MiniMaxCodingAccountsConfig {
    pub fn validate(&self) -> anyhow::Result<()> {
        anyhow::ensure!(
            self.schema == CONFIG_SCHEMA,
            "MiniMax config schema is invalid"
        );
        let mut ids = HashSet::new();
        let mut secrets = HashSet::new();
        for account in &self.accounts {
            account.validate()?;
            anyhow::ensure!(
                ids.insert(account.id.as_str()),
                "duplicate MiniMax account id"
            );
            anyhow::ensure!(
                secrets.insert((
                    account.api_key_secret.scope.trim(),
                    account.api_key_secret.name.trim()
                )),
                "MiniMax accounts must not alias one secret reference"
            );
        }
        Ok(())
    }
}

fn config_schema() -> String {
    CONFIG_SCHEMA.to_owned()
}

pub fn load_accounts(root: &Path) -> anyhow::Result<MiniMaxCodingAccountsConfig> {
    let config: MiniMaxCodingAccountsConfig =
        crate::persistence::load_json_payload(root, STORAGE_KEY)?.unwrap_or_default();
    config.validate()?;
    Ok(config)
}

pub fn store_accounts(root: &Path, config: &MiniMaxCodingAccountsConfig) -> anyhow::Result<()> {
    config.validate()?;
    crate::persistence::store_json_payload(root, STORAGE_KEY, Some(config))
}

/// Non-secret status projection suitable for Business OS and model capability
/// rendering. This performs only local encrypted-store metadata checks.
pub fn account_statuses(root: &Path) -> anyhow::Result<Vec<MiniMaxCodingAccountStatus>> {
    load_accounts(root)?
        .accounts
        .into_iter()
        .map(|account| {
            let models = account.effective_models();
            let phase = if account.disabled {
                MiniMaxCodingAccountPhase::Disabled
            } else if account.credential_is_ready(root)? {
                MiniMaxCodingAccountPhase::Ready
            } else {
                MiniMaxCodingAccountPhase::MissingCredential
            };
            Ok(MiniMaxCodingAccountStatus {
                id: account.id,
                phase,
                models,
                endpoint_profile: account.endpoint_profile,
            })
        })
        .collect()
}

/// Adds or updates one typed account without touching credential material.
/// Account identity remains explicit; no model/provider inference is used.
pub fn upsert_account(root: &Path, account: MiniMaxCodingAccount) -> anyhow::Result<()> {
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

/// Rotates only the selected account's encrypted secret record. The account
/// must already exist and the secret handle remains stable across rotation.
pub fn rotate_api_key(root: &Path, account_id: &str, api_key: &str) -> anyhow::Result<()> {
    let _lifecycle = crate::secrets::credential_lifecycle_guard();
    let api_key = api_key.trim();
    anyhow::ensure!(!api_key.is_empty(), "MiniMax API key is required");
    let config = load_accounts(root)?;
    let account = config
        .accounts
        .iter()
        .find(|account| account.id == account_id.trim())
        .with_context(|| format!("MiniMax coding account does not exist: {account_id}"))?;
    crate::secrets::write_secret_record(
        root,
        account.api_key_secret.scope.trim(),
        account.api_key_secret.name.trim(),
        api_key,
        Some("MiniMax Coding Plan API key".to_owned()),
        serde_json::json!({
            "provider": "minimax",
            "access_mode": "coding_plan",
            "account_id": account.id,
        }),
    )?;
    Ok(())
}

/// Disconnects one account fail-closed: topology is removed atomically from
/// CTOX's payload store before credential cleanup. A cleanup failure can leave
/// only an unreferenced encrypted orphan, never a still-advertised broken
/// account.
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
    delete_secret: impl FnOnce(&MiniMaxCodingAccount) -> anyhow::Result<()>,
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

pub fn ready_accounts(root: &Path) -> anyhow::Result<Vec<MiniMaxCodingAccount>> {
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

pub fn resolve_ready_account(
    root: &Path,
    account_id: &str,
) -> anyhow::Result<MiniMaxCodingAccount> {
    let account_id = account_id.trim();
    anyhow::ensure!(!account_id.is_empty(), "MiniMax account id is required");
    let mut matches = ready_accounts(root)?
        .into_iter()
        .filter(|account| account.id == account_id);
    let account = matches
        .next()
        .with_context(|| format!("MiniMax coding account is unavailable: {account_id}"))?;
    anyhow::ensure!(
        matches.next().is_none(),
        "MiniMax coding account is ambiguous"
    );
    Ok(account)
}

pub fn read_api_key(root: &Path, account: &MiniMaxCodingAccount) -> anyhow::Result<String> {
    crate::secrets::read_secret_value(
        root,
        account.api_key_secret.scope.trim(),
        account.api_key_secret.name.trim(),
    )
    .context("MiniMax coding credential is unavailable")
}

#[derive(Debug, Clone, PartialEq)]
pub struct MiniMaxCapacityMeasurement {
    pub used: f64,
    pub limit: f64,
    pub unit: String,
    pub reset_at_epoch_seconds: Option<u64>,
    pub rate_limited: bool,
}

/// Parses only explicitly named general/MiniMax/coding windows. Ambiguous
/// counters, media quotas and inconsistent used/limit pairs remain unknown.
pub fn parse_capacity(payload: &[u8]) -> Option<MiniMaxCapacityMeasurement> {
    let root: Value = serde_json::from_slice(payload).ok()?;
    if root
        .pointer("/base_resp/status_code")
        .and_then(json_number)
        .is_some_and(|code| code != 0.0)
    {
        return None;
    }
    let entries = root.get("model_remains")?.as_array()?;
    let mut windows = Vec::new();
    for entry in entries {
        let Some(raw_name) = entry.get("model_name").and_then(Value::as_str) else {
            continue;
        };
        let name = raw_name.to_ascii_lowercase();
        if name != "general" && !name.contains("minimax-m") && !name.contains("coding") {
            continue;
        }
        append_capacity_window(
            &mut windows,
            entry,
            raw_name,
            "interval",
            "current_interval_usage_count",
            "current_interval_total_count",
            "current_interval_remaining_percent",
            "end_time",
        );
        append_capacity_window(
            &mut windows,
            entry,
            raw_name,
            "weekly",
            "current_weekly_usage_count",
            "current_weekly_total_count",
            "current_weekly_remaining_percent",
            "weekly_end_time",
        );
    }
    windows.into_iter().max_by(|left, right| {
        (left.used / left.limit)
            .partial_cmp(&(right.used / right.limit))
            .unwrap_or(std::cmp::Ordering::Equal)
    })
}

/// Explicit, account-scoped capacity refresh. This is intentionally not run
/// during capability rendering or daemon polling: resolving an encrypted
/// credential and contacting the provider must follow an operator/runtime
/// action. MiniMax uses Bearer auth for this endpoint even though its
/// Anthropic-compatible inference endpoint uses `x-api-key`.
pub fn fetch_capacity(
    root: &Path,
    account_id: &str,
) -> anyhow::Result<Option<MiniMaxCapacityMeasurement>> {
    let account = resolve_ready_account(root, account_id)?;
    let api_key = read_api_key(root, &account)?;
    let response = match ureq::get(CAPACITY_URL)
        .set("accept", "application/json")
        .set("authorization", &format!("Bearer {api_key}"))
        .timeout(Duration::from_secs(20))
        .call()
    {
        Ok(response) => response,
        Err(ureq::Error::Status(_, response)) => response,
        Err(error) => return Err(anyhow::anyhow!(error).context("MiniMax capacity unavailable")),
    };
    anyhow::ensure!(
        (200..300).contains(&response.status()),
        "MiniMax capacity request was rejected with HTTP {}",
        response.status()
    );
    let mut payload = Vec::new();
    response
        .into_reader()
        .take(1_048_577)
        .read_to_end(&mut payload)
        .context("read MiniMax capacity response")?;
    anyhow::ensure!(
        payload.len() <= 1_048_576,
        "MiniMax capacity response is too large"
    );
    Ok(parse_capacity(&payload))
}

#[allow(clippy::too_many_arguments)]
fn append_capacity_window(
    windows: &mut Vec<MiniMaxCapacityMeasurement>,
    entry: &Value,
    model: &str,
    window: &str,
    used_key: &str,
    limit_key: &str,
    remaining_key: &str,
    reset_key: &str,
) {
    let counts = entry
        .get(used_key)
        .and_then(json_number)
        .zip(entry.get(limit_key).and_then(json_number))
        .filter(|(used, limit)| *used >= 0.0 && *limit > 0.0 && *used <= *limit);
    let (used, limit) = match counts {
        Some(values) => values,
        None => match entry.get(remaining_key).and_then(json_number) {
            Some(remaining) if (0.0..=100.0).contains(&remaining) => (100.0 - remaining, 100.0),
            _ => return,
        },
    };
    let reset_at_epoch_seconds = entry
        .get(reset_key)
        .and_then(json_number)
        .filter(|value| *value > 0.0)
        .map(|value| {
            if value > 10_000_000_000.0 {
                value / 1000.0
            } else {
                value
            }
        })
        .map(|value| value as u64);
    windows.push(MiniMaxCapacityMeasurement {
        used,
        limit,
        unit: format!("{model}:{window}"),
        reset_at_epoch_seconds,
        rate_limited: used >= limit,
    });
}

fn json_number(value: &Value) -> Option<f64> {
    value
        .as_f64()
        .or_else(|| value.as_str().and_then(|value| value.parse().ok()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn account(id: &str) -> MiniMaxCodingAccount {
        MiniMaxCodingAccount {
            id: id.to_owned(),
            disabled: false,
            models: Vec::new(),
            api_key_secret: MiniMaxSecretRef {
                scope: "provider-subscriptions".to_owned(),
                name: format!("{id}-api-key"),
            },
            endpoint_profile: MiniMaxCodingEndpointProfile::GlobalAnthropic,
        }
    }

    #[test]
    fn account_config_round_trips_without_secret_material() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let config = MiniMaxCodingAccountsConfig {
            schema: CONFIG_SCHEMA.to_owned(),
            accounts: vec![account("mini-primary")],
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
        Ok(())
    }

    #[test]
    fn readiness_requires_the_independent_secret_reference() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let config = MiniMaxCodingAccountsConfig {
            accounts: vec![account("mini-one")],
            ..MiniMaxCodingAccountsConfig::default()
        };
        store_accounts(temp.path(), &config)?;
        assert!(ready_accounts(temp.path())?.is_empty());
        crate::secrets::write_secret_record(
            temp.path(),
            "provider-subscriptions",
            "mini-one-api-key",
            "secret-value",
            None,
            serde_json::json!({"provider":"minimax","access_mode":"coding_plan"}),
        )?;
        assert_eq!(ready_accounts(temp.path())?[0].id, "mini-one");
        assert_eq!(
            read_api_key(temp.path(), &config.accounts[0])?,
            "secret-value"
        );
        Ok(())
    }

    #[test]
    fn status_rotation_and_disconnect_are_account_scoped_and_fail_closed() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        upsert_account(root, account("mini-one"))?;
        upsert_account(root, account("mini-two"))?;
        assert_eq!(
            account_statuses(root)?
                .iter()
                .map(|status| status.phase)
                .collect::<Vec<_>>(),
            [
                MiniMaxCodingAccountPhase::MissingCredential,
                MiniMaxCodingAccountPhase::MissingCredential,
            ]
        );

        rotate_api_key(root, "mini-one", "first")?;
        assert_eq!(
            account_statuses(root)?[0].phase,
            MiniMaxCodingAccountPhase::Ready
        );
        rotate_api_key(root, "mini-one", "second")?;
        assert_eq!(read_api_key(root, &account("mini-one"))?, "second");

        let connection = rusqlite::Connection::open(crate::secrets::secret_store_path(root))?;
        connection.execute_batch(
            r#"
            CREATE TRIGGER reject_minimax_coding_rotation
            BEFORE UPDATE ON ctox_secret_records
            WHEN NEW.secret_name = 'mini-one-api-key'
            BEGIN
                SELECT RAISE(ABORT, 'forced MiniMax Coding rollback');
            END;
            "#,
        )?;
        drop(connection);
        assert!(rotate_api_key(root, "mini-one", "rejected").is_err());
        assert_eq!(read_api_key(root, &account("mini-one"))?, "second");
        assert!(disconnect_account(root, "mini-one")?);
        assert!(!disconnect_account(root, "mini-one")?);
        assert!(!crate::secrets::secret_exists(
            root,
            "provider-subscriptions",
            "mini-one-api-key"
        )?);
        assert_eq!(load_accounts(root)?.accounts[0].id, "mini-two");
        Ok(())
    }

    #[test]
    fn disconnect_cleanup_failure_leaves_only_an_unreferenced_secret() -> anyhow::Result<()> {
        let temp = tempfile::tempdir()?;
        let root = temp.path();
        upsert_account(root, account("mini-orphan"))?;
        rotate_api_key(root, "mini-orphan", "secret")?;

        let error = disconnect_account_with(root, "mini-orphan", |_| {
            anyhow::bail!("synthetic cleanup failure")
        })
        .unwrap_err()
        .to_string();
        assert!(error.contains("synthetic cleanup failure"));
        assert!(load_accounts(root)?.accounts.is_empty());
        assert!(crate::secrets::secret_exists(
            root,
            "provider-subscriptions",
            "mini-orphan-api-key"
        )?);
        assert!(ready_accounts(root)?.is_empty());
        Ok(())
    }

    #[test]
    fn capacity_parser_keeps_worst_supported_window_and_rejects_media() {
        let parsed = parse_capacity(
            br#"{"base_resp":{"status_code":0},"model_remains":[{"model_name":"video-01","current_interval_total_count":10,"current_interval_usage_count":10},{"model_name":"general","current_interval_remaining_percent":65,"current_weekly_remaining_percent":20,"weekly_end_time":1786320000000}]}"#,
        )
        .expect("supported capacity");
        assert_eq!((parsed.used, parsed.limit), (80.0, 100.0));
        assert_eq!(parsed.unit, "general:weekly");
        assert_eq!(parsed.reset_at_epoch_seconds, Some(1_786_320_000));

        assert!(parse_capacity(
            br#"{"model_remains":[{"model_name":"image-01","current_interval_total_count":100,"current_interval_usage_count":10}]}"#
        )
        .is_none());
        assert!(parse_capacity(
            br#"{"model_remains":[{"model_name":"general","current_interval_total_count":100,"current_interval_usage_count":120}]}"#
        )
        .is_none());
    }
}

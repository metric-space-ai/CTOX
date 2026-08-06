// ref: internal/api/handlers/management/config_lists.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde::{Deserialize, Serialize};

use crate::internal::config::{
    AntigravitySubscriptionAccountConfig, ClaudeSubscriptionAccountConfig,
    CodexSubscriptionAccountConfig,
};

use super::{ManagementConfigError, ManagementConfigService};

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(tag = "provider", content = "account", rename_all = "kebab-case")]
pub enum ManagementAccountConfig {
    Claude(ClaudeSubscriptionAccountConfig),
    Codex(CodexSubscriptionAccountConfig),
    Antigravity(AntigravitySubscriptionAccountConfig),
}

impl ManagementAccountConfig {
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Claude(account) => &account.id,
            Self::Codex(account) => &account.id,
            Self::Antigravity(account) => &account.id,
        }
    }
}

impl ManagementConfigService {
    pub fn accounts(&self) -> Result<Vec<ManagementAccountConfig>, ManagementConfigError> {
        let config = self.snapshot()?;
        Ok(config
            .claude_accounts
            .into_iter()
            .map(ManagementAccountConfig::Claude)
            .chain(
                config
                    .codex_accounts
                    .into_iter()
                    .map(ManagementAccountConfig::Codex),
            )
            .chain(
                config
                    .antigravity_accounts
                    .into_iter()
                    .map(ManagementAccountConfig::Antigravity),
            )
            .collect())
    }

    pub fn upsert_account(
        &self,
        account: ManagementAccountConfig,
    ) -> Result<ManagementAccountConfig, ManagementConfigError> {
        let result = account.clone();
        self.mutate(move |config| match account {
            ManagementAccountConfig::Claude(account) => {
                upsert(&mut config.claude_accounts, account, |item| &item.id);
            }
            ManagementAccountConfig::Codex(account) => {
                upsert(&mut config.codex_accounts, account, |item| &item.id);
            }
            ManagementAccountConfig::Antigravity(account) => {
                upsert(&mut config.antigravity_accounts, account, |item| &item.id);
            }
        })?;
        Ok(result)
    }

    pub fn delete_account(
        &self,
        provider: &str,
        auth_id: &str,
    ) -> Result<(), ManagementConfigError> {
        let provider = provider.trim().to_ascii_lowercase();
        let auth_id = auth_id.trim().to_owned();
        let mut found = false;
        self.mutate(|config| {
            let accounts: &mut dyn RetainAccounts = match provider.as_str() {
                "claude" => &mut config.claude_accounts,
                "codex" => &mut config.codex_accounts,
                "antigravity" => &mut config.antigravity_accounts,
                _ => return,
            };
            found = accounts.remove_id(&auth_id);
        })?;
        if found {
            Ok(())
        } else {
            Err(ManagementConfigError::AccountNotFound)
        }
    }
}

fn upsert<T>(items: &mut Vec<T>, replacement: T, id: impl Fn(&T) -> &str) {
    let replacement_id = id(&replacement).trim();
    if let Some(index) = items
        .iter()
        .position(|item| id(item).trim() == replacement_id)
    {
        items[index] = replacement;
    } else {
        items.push(replacement);
    }
}

trait RetainAccounts {
    fn remove_id(&mut self, id: &str) -> bool;
}

macro_rules! impl_retain_accounts {
    ($account:ty) => {
        impl RetainAccounts for Vec<$account> {
            fn remove_id(&mut self, id: &str) -> bool {
                let previous = self.len();
                self.retain(|account| account.id.trim() != id);
                self.len() != previous
            }
        }
    };
}

impl_retain_accounts!(ClaudeSubscriptionAccountConfig);
impl_retain_accounts!(CodexSubscriptionAccountConfig);
impl_retain_accounts!(AntigravitySubscriptionAccountConfig);

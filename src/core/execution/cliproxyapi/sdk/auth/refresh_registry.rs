// ref: sdk/auth/refresh_registry.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use super::Authenticator;

pub type AuthenticatorFactory = Arc<dyn Fn() -> Arc<dyn Authenticator> + Send + Sync + 'static>;

/// Explicit SDK refresh-lead registry. Go package `init()` registration is
/// replaced by host construction so independent gateways cannot mutate each
/// other's provider policy.
#[derive(Clone, Default)]
pub struct RefreshLeadRegistry {
    factories: BTreeMap<String, AuthenticatorFactory>,
}

impl RefreshLeadRegistry {
    pub fn register(&mut self, provider: &str, factory: AuthenticatorFactory) {
        let provider = provider.trim().to_lowercase();
        if !provider.is_empty() {
            self.factories.insert(provider, factory);
        }
    }

    #[must_use]
    pub fn refresh_lead(&self, provider: &str) -> Option<Duration> {
        let factory = self.factories.get(&provider.trim().to_lowercase())?;
        factory().refresh_lead().filter(|lead| !lead.is_zero())
    }

    pub fn providers(&self) -> impl Iterator<Item = &str> {
        self.factories.keys().map(String::as_str)
    }
}

impl fmt::Debug for RefreshLeadRegistry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RefreshLeadRegistry")
            .field("providers", &self.factories.keys().collect::<Vec<_>>())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::auth::{
        AuthenticatorError, LoginCancellation, LoginConfig, LoginFuture, LoginOptions,
    };

    struct Fixed;

    impl Authenticator for Fixed {
        fn provider(&self) -> &str {
            "fixed"
        }

        fn login<'a>(
            &'a self,
            _cancellation: &'a LoginCancellation,
            _config: &'a LoginConfig,
            _options: &'a LoginOptions,
        ) -> LoginFuture<'a> {
            Box::pin(async {
                Err(AuthenticatorError::new(
                    super::super::AuthenticatorErrorKind::LoginFailed,
                ))
            })
        }

        fn refresh_lead(&self) -> Option<Duration> {
            Some(Duration::from_secs(300))
        }
    }

    #[test]
    fn registry_normalizes_provider_and_reads_fresh_authenticator() {
        let mut registry = RefreshLeadRegistry::default();
        registry.register(" KIMI ", Arc::new(|| Arc::new(Fixed)));
        assert_eq!(registry.providers().collect::<Vec<_>>(), vec!["kimi"]);
        assert_eq!(
            registry.refresh_lead("Kimi"),
            Some(Duration::from_secs(300))
        );
        assert_eq!(registry.refresh_lead("missing"), None);
    }
}

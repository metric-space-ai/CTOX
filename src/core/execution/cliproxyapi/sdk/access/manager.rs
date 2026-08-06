// ref: sdk/access/manager.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::RwLock;

use super::{
    is_auth_error_code, new_invalid_credential_error, new_no_credentials_error,
    AuthenticationOutcome, Request, SharedProvider, AUTH_ERROR_CODE_INVALID_CREDENTIAL,
    AUTH_ERROR_CODE_NOT_HANDLED, AUTH_ERROR_CODE_NO_CREDENTIALS,
};

/// Coordinates request-authentication providers in a stable snapshot order.
///
/// Optional entries preserve nil Go interface slots. Provider futures run
/// after the snapshot lock is released, so authentication may safely trigger
/// a concurrent manager reconfiguration without deadlocking the request.
#[derive(Default)]
pub struct Manager {
    providers: RwLock<Vec<Option<SharedProvider>>>,
}

impl Manager {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    pub fn set_providers(&self, providers: &[Option<SharedProvider>]) {
        *self
            .providers
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = providers.to_vec();
    }

    pub fn set_shared_providers(&self, providers: &[SharedProvider]) {
        self.set_providers(
            &providers
                .iter()
                .map(|provider| Some(provider.clone()))
                .collect::<Vec<_>>(),
        );
    }

    #[must_use]
    pub fn providers(&self) -> Vec<Option<SharedProvider>> {
        self.providers
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    /// Evaluates providers until one succeeds or returns a terminal error.
    pub async fn authenticate(&self, request: &mut Request) -> AuthenticationOutcome {
        let providers = self.providers();
        if providers.is_empty() {
            return AuthenticationOutcome::default();
        }

        let mut invalid = false;
        for provider in providers.into_iter().flatten() {
            let outcome = provider.authenticate(request).await;
            let Some(error) = outcome.error else {
                return AuthenticationOutcome::success(outcome.result);
            };
            if is_auth_error_code(Some(&error), &AUTH_ERROR_CODE_NOT_HANDLED) {
                continue;
            }
            if is_auth_error_code(Some(&error), &AUTH_ERROR_CODE_NO_CREDENTIALS) {
                continue;
            }
            if is_auth_error_code(Some(&error), &AUTH_ERROR_CODE_INVALID_CREDENTIAL) {
                invalid = true;
                continue;
            }
            return AuthenticationOutcome::failure(error);
        }

        if invalid {
            AuthenticationOutcome::failure(new_invalid_credential_error())
        } else {
            // Upstream returns the same missing-credentials error both when a
            // provider reported it and when every provider declined to handle
            // the request, so no separate `missing` flag is observable here.
            AuthenticationOutcome::failure(new_no_credentials_error())
        }
    }
}

impl fmt::Debug for Manager {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AccessManager")
            .field("provider_count", &self.providers().len())
            .finish_non_exhaustive()
    }
}

/// Rust equivalent of invoking `Providers` on a possibly nil Go manager.
#[must_use]
pub fn manager_providers(manager: Option<&Manager>) -> Option<Vec<Option<SharedProvider>>> {
    manager.map(Manager::providers)
}

/// Rust equivalent of invoking `Authenticate` on a possibly nil Go manager.
pub async fn manager_authenticate(
    manager: Option<&Manager>,
    request: &mut Request,
) -> AuthenticationOutcome {
    match manager {
        Some(manager) => manager.authenticate(request).await,
        None => AuthenticationOutcome::default(),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use super::*;
    use crate::sdk::access::{
        new_internal_auth_error, new_not_handled_error, AuthError, AuthenticationFuture, Provider,
        Result, AUTH_ERROR_CODE_INVALID_CREDENTIAL, AUTH_ERROR_CODE_NO_CREDENTIALS,
    };

    #[derive(Clone, Copy)]
    enum Behavior {
        Success,
        NotHandled,
        Missing,
        Invalid,
        Internal,
    }

    struct TestProvider {
        id: &'static str,
        behavior: Behavior,
        mutation: Option<(&'static str, &'static str)>,
    }

    impl TestProvider {
        fn shared(id: &'static str, behavior: Behavior) -> SharedProvider {
            Arc::new(Self {
                id,
                behavior,
                mutation: None,
            })
        }

        fn mutating(
            id: &'static str,
            behavior: Behavior,
            key: &'static str,
            value: &'static str,
        ) -> SharedProvider {
            Arc::new(Self {
                id,
                behavior,
                mutation: Some((key, value)),
            })
        }
    }

    impl Provider for TestProvider {
        fn identifier(&self) -> &str {
            self.id
        }

        fn authenticate<'a>(&'a self, request: &'a mut Request) -> AuthenticationFuture<'a> {
            Box::pin(async move {
                if let Some((key, value)) = self.mutation {
                    request
                        .headers
                        .get_or_insert_with(BTreeMap::new)
                        .insert(key.to_owned(), vec![value.to_owned()]);
                }
                match self.behavior {
                    Behavior::Success => AuthenticationOutcome::success(Some(Result {
                        provider: self.id.to_owned(),
                        principal: self.id.to_owned(),
                        metadata: None,
                    })),
                    Behavior::NotHandled => AuthenticationOutcome::failure(new_not_handled_error()),
                    Behavior::Missing => AuthenticationOutcome::failure(AuthError {
                        code: AUTH_ERROR_CODE_NO_CREDENTIALS,
                        message: "provider-specific missing".to_owned(),
                        status_code: 401,
                        cause: None,
                    }),
                    Behavior::Invalid => AuthenticationOutcome::failure(AuthError {
                        code: AUTH_ERROR_CODE_INVALID_CREDENTIAL,
                        message: "provider-specific invalid".to_owned(),
                        status_code: 401,
                        cause: None,
                    }),
                    Behavior::Internal => AuthenticationOutcome::failure(new_internal_auth_error(
                        "provider exploded",
                        None,
                    )),
                }
            })
        }
    }

    #[tokio::test]
    async fn empty_and_nil_manager_preserve_upstream_no_decision() {
        let mut request = Request::default();
        let empty = Manager::new().authenticate(&mut request).await;
        assert!(empty.result.is_none());
        assert!(empty.error.is_none());

        let nil = manager_authenticate(None, &mut request).await;
        assert!(nil.result.is_none());
        assert!(nil.error.is_none());
        assert!(manager_providers(None).is_none());
        assert_eq!(manager_providers(Some(&Manager::new())).unwrap().len(), 0);
    }

    #[tokio::test]
    async fn not_handled_and_nil_slots_continue_to_first_success() {
        let manager = Manager::new();
        manager.set_providers(&[
            None,
            Some(TestProvider::shared("skip", Behavior::NotHandled)),
            Some(TestProvider::shared("winner", Behavior::Success)),
            Some(TestProvider::shared("late", Behavior::Internal)),
        ]);
        let outcome = manager.authenticate(&mut Request::default()).await;
        assert!(outcome.error.is_none());
        assert_eq!(outcome.result.unwrap().provider, "winner");
    }

    #[tokio::test]
    async fn invalid_dominates_missing_and_terminal_error_returns_immediately() {
        let manager = Manager::new();
        manager.set_shared_providers(&[
            TestProvider::shared("missing", Behavior::Missing),
            TestProvider::shared("invalid", Behavior::Invalid),
            TestProvider::shared("skip", Behavior::NotHandled),
        ]);
        let outcome = manager.authenticate(&mut Request::default()).await;
        assert_eq!(
            outcome.error.unwrap().code,
            AUTH_ERROR_CODE_INVALID_CREDENTIAL
        );

        manager.set_shared_providers(&[
            TestProvider::shared("internal", Behavior::Internal),
            TestProvider::shared("winner", Behavior::Success),
        ]);
        let outcome = manager.authenticate(&mut Request::default()).await;
        assert_eq!(outcome.error.unwrap().message, "provider exploded");
    }

    #[tokio::test]
    async fn all_skipped_defaults_to_missing_and_request_mutations_reach_next_provider() {
        let manager = Manager::new();
        manager.set_shared_providers(&[
            TestProvider::mutating("skip", Behavior::NotHandled, "X-Test", "visible"),
            TestProvider::shared("winner", Behavior::Success),
        ]);
        let mut request = Request::default();
        let outcome = manager.authenticate(&mut request).await;
        assert_eq!(outcome.result.unwrap().provider, "winner");
        assert_eq!(request.headers.unwrap()["X-Test"], ["visible".to_owned()]);

        manager.set_providers(&[
            None,
            Some(TestProvider::shared("skip", Behavior::NotHandled)),
        ]);
        let outcome = manager.authenticate(&mut Request::default()).await;
        assert_eq!(outcome.error.unwrap().code, AUTH_ERROR_CODE_NO_CREDENTIALS);
    }

    #[test]
    fn snapshots_preserve_arc_identity_nil_slots_and_replacement_isolated() {
        let manager = Manager::new();
        let provider = TestProvider::shared("one", Behavior::Success);
        manager.set_providers(&[None, Some(provider.clone())]);
        let snapshot = manager.providers();
        assert!(snapshot[0].is_none());
        assert!(Arc::ptr_eq(snapshot[1].as_ref().unwrap(), &provider));

        manager.set_providers(&[]);
        assert!(manager.providers().is_empty());
        assert!(snapshot[1].is_some());
    }
}

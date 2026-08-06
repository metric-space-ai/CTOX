// ref: sdk/access/registry.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock, RwLock};

use super::AuthError;

pub type Headers = BTreeMap<String, Vec<String>>;

/// Owned incoming-request view used by access providers.
///
/// The Go interface receives `*http.Request`. The Rust boundary owns the body
/// so an adapter can inspect and restore it without an `io.ReadCloser`; dropped
/// provider futures supply Rust's cancellation semantics in place of
/// `context.Context`. Optional fields preserve nil URL/header/body values.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Request {
    pub method: String,
    pub url: Option<String>,
    pub headers: Option<Headers>,
    pub body: Option<Vec<u8>>,
}

impl Request {
    #[must_use]
    pub fn new(method: impl Into<String>, url: impl Into<String>) -> Self {
        Self {
            method: method.into(),
            url: Some(url.into()),
            ..Self::default()
        }
    }
}

/// Authentication outcome returned by a provider.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Result {
    pub provider: String,
    pub principal: String,
    /// `None` is a nil Go map; `Some(empty)` is a non-nil empty map.
    pub metadata: Option<BTreeMap<String, String>>,
}

/// The two optional values deliberately are not a Rust `Result`: Go permits
/// `(nil, nil)` and can technically return both a result and an error. Keeping
/// them independent lets the later manager port reproduce that contract.
#[derive(Clone, Debug, Default)]
pub struct AuthenticationOutcome {
    pub result: Option<Result>,
    pub error: Option<AuthError>,
}

impl AuthenticationOutcome {
    #[must_use]
    pub fn success(result: Option<Result>) -> Self {
        Self {
            result,
            error: None,
        }
    }

    #[must_use]
    pub fn failure(error: AuthError) -> Self {
        Self {
            result: None,
            error: Some(error),
        }
    }
}

pub type AuthenticationFuture<'a> =
    Pin<Box<dyn Future<Output = AuthenticationOutcome> + Send + 'a>>;

/// Validates credentials for incoming requests.
pub trait Provider: Send + Sync {
    fn identifier(&self) -> &str;

    fn authenticate<'a>(&'a self, request: &'a mut Request) -> AuthenticationFuture<'a>;
}

pub type SharedProvider = Arc<dyn Provider>;

#[derive(Default)]
struct RegistryState {
    registry: BTreeMap<String, SharedProvider>,
    order: Vec<String>,
    exclusive_provider: String,
}

fn registry_state() -> &'static RwLock<RegistryState> {
    static REGISTRY: OnceLock<RwLock<RegistryState>> = OnceLock::new();
    REGISTRY.get_or_init(|| RwLock::new(RegistryState::default()))
}

/// Registers a pre-built provider instance for a type identifier.
///
/// `None` is the Rust representation of Go's nil provider and is ignored.
/// Replacing an existing key preserves its original registration position.
pub fn register_provider(provider_type: &str, provider: Option<SharedProvider>) {
    let normalized_type = provider_type.trim();
    let Some(provider) = provider else {
        return;
    };
    if normalized_type.is_empty() {
        return;
    }

    let mut state = registry_state()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if !state.registry.contains_key(normalized_type) {
        state.order.push(normalized_type.to_owned());
    }
    state.registry.insert(normalized_type.to_owned(), provider);
}

/// Removes a provider by type identifier.
pub fn unregister_provider(provider_type: &str) {
    let normalized_type = provider_type.trim();
    if normalized_type.is_empty() {
        return;
    }

    let mut state = registry_state()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.registry.remove(normalized_type).is_none() {
        return;
    }
    if let Some(index) = state
        .order
        .iter()
        .position(|registered_type| registered_type == normalized_type)
    {
        state.order.remove(index);
    }
}

/// Restricts snapshots to one provider key when that key is present.
pub fn set_exclusive_provider(provider_type: &str) {
    let mut state = registry_state()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.exclusive_provider = provider_type.trim().to_owned();
}

/// Removes any active provider restriction.
pub fn clear_exclusive_provider() {
    let mut state = registry_state()
        .write()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    state.exclusive_provider.clear();
}

/// Returns provider instances in registration order.
///
/// `None` preserves upstream's nil slice when no provider has ever remained in
/// the registry. An exclusive key only restricts the result while that key is
/// present; a stale key deliberately falls back to the full ordered snapshot.
#[must_use]
pub fn registered_providers() -> Option<Vec<SharedProvider>> {
    let state = registry_state()
        .read()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    if state.order.is_empty() {
        return None;
    }
    if !state.exclusive_provider.is_empty() {
        if let Some(provider) = state.registry.get(&state.exclusive_provider) {
            return Some(vec![Arc::clone(provider)]);
        }
    }
    Some(
        state
            .order
            .iter()
            .filter_map(|provider_type| state.registry.get(provider_type).map(Arc::clone))
            .collect(),
    )
}

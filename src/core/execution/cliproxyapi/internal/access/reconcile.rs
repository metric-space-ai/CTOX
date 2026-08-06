// ref: internal/access/reconcile.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use crate::sdk::access::{
    registered_providers, Manager, SharedProvider, DEFAULT_ACCESS_PROVIDER_NAME,
};

/// Reconciles the current registry snapshot with an existing provider list.
/// Registry order is retained and change lists are sorted like upstream.
#[must_use]
pub fn reconcile_providers(
    existing: &[Option<SharedProvider>],
) -> (Vec<SharedProvider>, Vec<String>, Vec<String>, Vec<String>) {
    let result = registered_providers().unwrap_or_default();
    reconcile_provider_snapshots(result, existing)
}

fn reconcile_provider_snapshots(
    result: Vec<SharedProvider>,
    existing: &[Option<SharedProvider>],
) -> (Vec<SharedProvider>, Vec<String>, Vec<String>, Vec<String>) {
    let existing_map: BTreeMap<_, _> = existing
        .iter()
        .flatten()
        .filter_map(|provider| {
            let id = provider.identifier().trim();
            (!id.is_empty()).then(|| (id.to_owned(), Arc::clone(provider)))
        })
        .collect();
    let final_ids: BTreeSet<_> = result
        .iter()
        .filter_map(|provider| {
            let id = provider.identifier().trim();
            (!id.is_empty()).then(|| id.to_owned())
        })
        .collect();

    let mut added = Vec::new();
    let mut updated = Vec::new();
    for provider in &result {
        let id = provider.identifier().trim();
        if id.is_empty() || id.eq_ignore_ascii_case(DEFAULT_ACCESS_PROVIDER_NAME) {
            continue;
        }
        match existing_map.get(id) {
            None => added.push(id.to_owned()),
            Some(existing) if !Arc::ptr_eq(existing, provider) => updated.push(id.to_owned()),
            Some(_) => {}
        }
    }

    let mut removed: Vec<_> = existing_map
        .keys()
        .filter(|id| {
            !final_ids.contains(*id) && !id.eq_ignore_ascii_case(DEFAULT_ACCESS_PROVIDER_NAME)
        })
        .cloned()
        .collect();
    added.sort();
    updated.sort();
    removed.sort();
    (result, added, updated, removed)
}

/// Registers the typed inline-key snapshot, reconciles it against the manager
/// and returns whether any externally visible provider changed.
pub fn apply_access_providers(manager: Option<&Manager>, api_keys: &[String]) -> bool {
    let Some(manager) = manager else {
        return false;
    };
    let existing = manager.providers();
    super::config_access::register(api_keys);
    let (providers, added, updated, removed) = reconcile_providers(&existing);
    manager.set_shared_providers(&providers);
    !(added.is_empty() && updated.is_empty() && removed.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sdk::access::{AuthenticationFuture, AuthenticationOutcome, Provider, Request};

    struct TestProvider(String);

    impl TestProvider {
        fn shared(id: &str) -> SharedProvider {
            Arc::new(Self(id.to_owned()))
        }
    }

    impl Provider for TestProvider {
        fn identifier(&self) -> &str {
            &self.0
        }

        fn authenticate<'a>(&'a self, _request: &'a mut Request) -> AuthenticationFuture<'a> {
            Box::pin(async { AuthenticationOutcome::default() })
        }
    }

    #[test]
    fn reconcile_sorts_changes_and_uses_provider_identity() {
        let old_a = TestProvider::shared("a");
        let removed = TestProvider::shared("z");
        let desired = vec![TestProvider::shared("a"), TestProvider::shared("b")];

        let (_, added, updated, removed_ids) =
            reconcile_provider_snapshots(desired, &[Some(old_a), Some(removed)]);

        assert_eq!(added, ["b"]);
        assert_eq!(updated, ["a"]);
        assert_eq!(removed_ids, ["z"]);
    }

    #[test]
    fn inline_provider_is_excluded_from_change_signal() {
        let desired = vec![TestProvider::shared(DEFAULT_ACCESS_PROVIDER_NAME)];
        let (_, added, updated, removed) = reconcile_provider_snapshots(desired, &[]);
        assert!(added.is_empty());
        assert!(updated.is_empty());
        assert!(removed.is_empty());
    }
}

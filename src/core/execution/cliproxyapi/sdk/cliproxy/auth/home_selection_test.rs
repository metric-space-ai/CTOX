// ref: sdk/cliproxy/auth/home_selection_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: instance-owned scope, resource teardown and retained route adaptation
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::sdk::cliproxy::executionregistry::{Registry, ScopeSpec};
use crate::sdk::pluginapi::ProviderExecutor;

use super::{Auth, HomeDispatchSelection};

pub(super) fn selection() -> (Arc<HomeDispatchSelection>, Arc<Registry>) {
    let registry = Arc::new(Registry::new());
    let pending = registry.begin_dispatch().unwrap();
    let scope = registry
        .install(
            &pending,
            ScopeSpec {
                request_id: "request".into(),
                credential_id: "auth".into(),
                model: "gpt-5".into(),
                ..ScopeSpec::default()
            },
        )
        .unwrap();
    let executor: Arc<dyn ProviderExecutor> =
        super::home_execution_paths_test::TestExecutor::failing(0);
    let mut auth = Auth::default();
    auth.id = "auth".into();
    auth.index = "auth".into();
    auth.provider = "codex".into();
    auth.attributes
        .insert("home_upstream_model".into(), "gpt-5(high)".into());
    auth.attributes
        .insert("home_force_mapping".into(), "true".into());
    (
        HomeDispatchSelection::new(auth, executor, " CoDeX ", scope).unwrap(),
        registry,
    )
}

#[test]
fn selection_closes_bound_resources_once_in_reverse_order() {
    let (selection, _registry) = selection();
    let order = Arc::new(std::sync::Mutex::new(Vec::new()));
    for value in [1, 2] {
        let order = order.clone();
        selection
            .bind(move || {
                order.lock().unwrap().push(value);
                Ok(())
            })
            .unwrap();
    }
    selection.end("done");
    selection.end("again");
    assert_eq!(*order.lock().unwrap(), vec![2, 1]);
    assert!(!selection.active());
}

#[test]
fn retained_selection_adapts_reasoning_suffix_and_force_alias() {
    let (selection, _registry) = selection();
    selection.retain();
    let auth = selection.clone_auth_for_route("public-model(max)");
    assert_eq!(auth.attributes["home_upstream_model"], "gpt-5(max)");
    assert_eq!(auth.attributes["home_original_alias"], "public-model(max)");
    selection.end("done");
}

#[test]
fn binding_after_end_closes_immediately_and_rejects_ownership() {
    let (selection, _registry) = selection();
    selection.end("done");
    let closed = Arc::new(AtomicUsize::new(0));
    let observed = closed.clone();
    assert!(selection
        .bind(move || {
            observed.fetch_add(1, Ordering::AcqRel);
            Ok(())
        })
        .is_err());
    assert_eq!(closed.load(Ordering::Acquire), 1);
}

#[test]
fn replace_auth_preserves_routing_attributes() {
    let (selection, _registry) = selection();
    let mut refreshed = Auth::default();
    refreshed.id = "auth".into();
    refreshed.provider = "codex".into();
    refreshed
        .metadata
        .insert("access_token".into(), serde_json::json!("fresh"));
    selection.replace_auth(refreshed);

    let updated = selection.clone_auth();
    assert_eq!(updated.metadata["access_token"], "fresh");
    assert_eq!(updated.attributes["home_upstream_model"], "gpt-5(high)");
    assert_eq!(updated.attributes["home_force_mapping"], "true");
}

#[test]
fn replace_auth_and_clone_are_concurrent_snapshot_operations() {
    let (selection, _registry) = selection();
    let writer = selection.clone();
    let thread = std::thread::spawn(move || {
        for _ in 0..1_000 {
            let mut auth = Auth::default();
            auth.id = "auth".into();
            writer.replace_auth(auth);
        }
    });
    for _ in 0..1_000 {
        assert_eq!(selection.clone_auth().id, "auth");
    }
    thread.join().unwrap();
}

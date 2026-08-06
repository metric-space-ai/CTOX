// ref: internal/api/handlers/management/plugins_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::json;

use super::test_store_test::{Runtime, Store};
use super::{
    ManagementPluginConfig, ManagementPluginConfigPatch, ManagementPluginError,
    ManagementPluginRuntimeRecord, ManagementPluginService,
};

fn service() -> (Arc<Store>, Arc<Runtime>, ManagementPluginService) {
    let store = Arc::new(Store::default());
    let runtime = Arc::new(Runtime::default());
    let service = ManagementPluginService::new(store.clone(), runtime.clone());
    (store, runtime, service)
}

#[test]
fn list_merges_configured_and_registered_plugins_deterministically() {
    let (_, runtime, service) = service();
    service.set_enabled("configured", true).unwrap();
    runtime
        .0
        .lock()
        .unwrap()
        .push(ManagementPluginRuntimeRecord {
            id: "registered".to_owned(),
            registered: true,
            supports_oauth: true,
            oauth_provider: Some("plugin-auth".to_owned()),
            capabilities: vec!["executor".to_owned()],
            install_ref: Some("plugin:registered@1".to_owned()),
        });
    let views = service.list().unwrap();
    assert_eq!(
        views
            .iter()
            .map(|view| view.id.as_str())
            .collect::<Vec<_>>(),
        ["configured", "registered"]
    );
    assert!(views[0].configured && views[0].enabled && !views[0].registered);
    assert!(views[1].registered && views[1].supports_oauth && !views[1].configured);
}

#[test]
fn effective_enabled_requires_global_config_instance_and_registration() {
    let (store, runtime, service) = service();
    store.0.lock().unwrap().plugins_enabled = true;
    service.set_enabled("sample", true).unwrap();
    runtime
        .0
        .lock()
        .unwrap()
        .push(ManagementPluginRuntimeRecord {
            id: "sample".to_owned(),
            registered: true,
            ..Default::default()
        });
    assert!(service.list().unwrap()[0].effective_enabled);
}

#[test]
fn patch_merges_and_deletes_only_named_public_fields() {
    let (_, _, service) = service();
    service
        .put_config(
            "sample",
            ManagementPluginConfig {
                enabled: true,
                priority: 7,
                values: BTreeMap::from([
                    ("mode".to_owned(), json!("safe")),
                    ("retries".to_owned(), json!(2)),
                ]),
            },
        )
        .unwrap();
    let patched = service
        .patch_config(
            "sample",
            ManagementPluginConfigPatch {
                priority: Some(9),
                values: BTreeMap::from([
                    ("mode".to_owned(), None),
                    ("strict".to_owned(), Some(json!(true))),
                ]),
                ..Default::default()
            },
        )
        .unwrap();
    assert_eq!(patched.priority, 9);
    assert!(!patched.values.contains_key("mode"));
    assert_eq!(patched.values["retries"], json!(2));
    assert_eq!(patched.values["strict"], json!(true));
}

#[test]
fn config_rejects_secret_shaped_fields_and_redacts_debug() {
    let (_, _, service) = service();
    let config = ManagementPluginConfig {
        values: BTreeMap::from([("api_key".to_owned(), json!("live-secret"))]),
        ..Default::default()
    };
    assert_eq!(
        service.put_config("sample", config.clone()),
        Err(ManagementPluginError::InvalidConfig)
    );
    let debug = format!("{config:?}");
    assert!(debug.contains("[REDACTED]"));
    assert!(!debug.contains("live-secret"));
}

#[test]
fn invalid_ids_never_reach_the_store() {
    let (store, _, service) = service();
    assert_eq!(
        service.set_enabled("../sample", true),
        Err(ManagementPluginError::InvalidPluginId)
    );
    assert_eq!(store.0.lock().unwrap().revision, 0);
}

// ref: sdk/cliproxy/auth/persist_policy_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::json;

use super::{should_persist, Auth, PersistenceIntent};

fn durable_auth() -> Auth {
    let mut auth = Auth::default();
    auth.id = "oauth".into();
    auth.provider = "antigravity".into();
    auth.metadata.insert("access_token".into(), json!("secret"));
    auth
}

#[test]
fn typed_source_already_persisted_intent_prevents_writeback() {
    let auth = durable_auth();
    assert!(should_persist(&auth, PersistenceIntent::Persist));
    assert!(!should_persist(
        &auth,
        PersistenceIntent::SourceAlreadyPersisted
    ));
}

#[test]
fn config_memory_and_plugin_virtual_records_never_enter_auth_store() {
    let mut config = durable_auth();
    config
        .attributes
        .insert("source".into(), "config:claude[0]".into());
    assert!(!should_persist(&config, PersistenceIntent::Persist));

    let mut runtime = durable_auth();
    runtime
        .attributes
        .insert("runtime_only".into(), "TRUE".into());
    assert!(!should_persist(&runtime, PersistenceIntent::Persist));

    let mut plugin = durable_auth();
    plugin.mark_plugin_virtual("plugin.toml", 1);
    assert!(!should_persist(&plugin, PersistenceIntent::Persist));
}

#[test]
fn empty_metadata_has_no_durable_credential_authority() {
    let mut auth = durable_auth();
    auth.metadata.clear();
    assert!(!should_persist(&auth, PersistenceIntent::Persist));
}

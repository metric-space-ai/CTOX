// ref: sdk/cliproxy/auth/request_auth_prepare_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: execution-local auth snapshots preserve metadata without mutating manager/Home auth
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use crate::sdk::pluginapi::ExecutorRequest;

use super::{prepare_executor_request, Auth};

#[test]
fn prepare_copies_all_auth_authority_into_owned_request_snapshot() {
    let mut auth = Auth::default();
    auth.id = "selected".into();
    auth.metadata
        .insert("access_token".into(), serde_json::json!("secret"));
    auth.attributes.insert("tenant".into(), "one".into());
    auth.attributes
        .insert("home_upstream_model".into(), "upstream-model".into());
    let original = ExecutorRequest {
        model: "public-model".into(),
        payload: b"body".to_vec(),
        ..Default::default()
    };
    let mut prepared = prepare_executor_request(&original, &auth, "codex");
    assert_eq!(prepared.auth_id, "selected");
    assert_eq!(prepared.auth_provider, "codex");
    assert_eq!(prepared.model, "upstream-model");
    assert_eq!(prepared.auth_metadata["access_token"], "secret");
    assert!(!prepared.storage_json.is_empty());
    prepared
        .auth_attributes
        .insert("tenant".into(), "mutated".into());
    assert_eq!(auth.attributes["tenant"], "one");
    assert_eq!(original.model, "public-model");
}

#[test]
fn preparation_does_not_mutate_same_id_local_auth() {
    let mut home = Auth::default();
    home.id = "same".into();
    home.metadata = BTreeMap::from([("token".into(), serde_json::json!("home"))]);
    let mut local = Auth::default();
    local.id = "same".into();
    local.metadata = BTreeMap::from([("token".into(), serde_json::json!("local"))]);
    let prepared = prepare_executor_request(&ExecutorRequest::default(), &home, "codex");
    assert_eq!(prepared.auth_metadata["token"], "home");
    assert_eq!(local.metadata["token"], "local");
}

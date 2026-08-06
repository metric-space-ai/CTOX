// ref: sdk/cliproxy/service_models_config_index_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::auth::Auth;
use super::service_models::{
    config_entry_for_auth_index, ConfiguredModel, ProviderKeyConfig, ATTRIBUTE_CONFIG_INDEX,
};
use std::collections::BTreeMap;

#[test]
fn config_index_selects_duplicate_named_or_credential_entries() {
    let entries = vec![
        ProviderKeyConfig {
            models: vec![ConfiguredModel {
                name: "first".into(),
                ..ConfiguredModel::default()
            }],
            ..ProviderKeyConfig::default()
        },
        ProviderKeyConfig {
            models: vec![ConfiguredModel {
                name: "second".into(),
                ..ConfiguredModel::default()
            }],
            ..ProviderKeyConfig::default()
        },
    ];
    let mut auth = Auth::default();
    auth.attributes = BTreeMap::from([
        ("source".into(), "config:claude[token]".into()),
        (ATTRIBUTE_CONFIG_INDEX.into(), "1".into()),
    ]);
    assert_eq!(
        config_entry_for_auth_index(&auth, &entries).unwrap().models[0].name,
        "second"
    );
}

#[test]
fn non_config_and_out_of_range_indexes_fail_closed() {
    let entries = [1_u8];
    let mut auth = Auth::default();
    auth.attributes = BTreeMap::from([
        ("source".into(), "file".into()),
        (ATTRIBUTE_CONFIG_INDEX.into(), "0".into()),
    ]);
    assert!(config_entry_for_auth_index(&auth, &entries).is_none());
}

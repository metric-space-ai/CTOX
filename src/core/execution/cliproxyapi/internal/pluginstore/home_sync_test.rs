// ref: internal/pluginstore/home_sync_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use chrono::{Duration, TimeZone, Utc};

use crate::sdk::pluginstore::{
    Manifest, PluginSyncItem, PluginSyncResponse, ResolvedAuthConfig, Secret, AUTH_TYPE_BEARER,
    PLUGIN_SYNC_SCHEMA_VERSION,
};

#[test]
fn sync_plan_validates_and_zeroizes_owned_secrets() {
    let now = Utc.with_ymd_and_hms(2026, 8, 4, 0, 0, 0).unwrap();
    let mut response = PluginSyncResponse {
        schema_version: PLUGIN_SYNC_SCHEMA_VERSION,
        expires_at: Some(now + Duration::minutes(5)),
        items: vec![PluginSyncItem {
            manifest: Manifest {
                id: "sample".into(),
                name: "Sample".into(),
                description: "Plugin".into(),
                author: "Author".into(),
                version: "1.0.0".into(),
                release_tag: "v1.0.0".into(),
                repository: "https://github.com/acme/sample".into(),
                ..Manifest::default()
            },
            auth: vec![ResolvedAuthConfig {
                match_url: "https://api.github.com/repos/acme/sample/".into(),
                auth_type: AUTH_TYPE_BEARER.into(),
                token: Secret::new(b"secret".to_vec()),
                ..ResolvedAuthConfig::default()
            }],
        }],
    };
    response.validate(now).unwrap();
    response.clear();
    assert!(response.items.is_empty());
    assert_eq!(response.schema_version, 0);
}

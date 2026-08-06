// ref: internal/pluginstore/auth_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use chrono::{TimeZone, Utc};

use crate::sdk::pluginstore::{
    ResolvedAuthConfig, Secret, AUTH_TYPE_BEARER, REQUEST_KIND_ARTIFACT,
};

use super::auth::{apply_resolved_auth, request_url_allowed, UrlPolicy};

#[test]
fn resolved_auth_matches_origin_and_path_boundaries() {
    let auth = vec![ResolvedAuthConfig {
        match_url: "https://downloads.example/private".to_owned(),
        apply_to: vec![REQUEST_KIND_ARTIFACT.to_owned()],
        auth_type: AUTH_TYPE_BEARER.to_owned(),
        token: Secret::new(b"secret-token".to_vec()),
        ..ResolvedAuthConfig::default()
    }];
    let now = Utc.with_ymd_and_hms(2026, 8, 4, 0, 0, 0).unwrap();
    for (url, expected) in [
        ("https://downloads.example/private", true),
        ("https://downloads.example/private/file.zip", true),
        ("https://downloads.example/private2/file.zip", false),
        ("https://downloads.example.evil/private/file.zip", false),
    ] {
        let mut headers = BTreeMap::new();
        assert_eq!(
            apply_resolved_auth(&mut headers, &auth, None, url, REQUEST_KIND_ARTIFACT, now)
                .unwrap(),
            expected
        );
        assert_eq!(headers.contains_key("Authorization"), expected);
    }
}

#[test]
fn expired_auth_and_ambient_http_are_rejected() {
    let now = Utc.with_ymd_and_hms(2026, 8, 4, 0, 0, 0).unwrap();
    let auth = vec![ResolvedAuthConfig {
        match_url: "https://downloads.example/".to_owned(),
        auth_type: AUTH_TYPE_BEARER.to_owned(),
        token: Secret::new(b"secret-token".to_vec()),
        ..ResolvedAuthConfig::default()
    }];
    let mut headers = BTreeMap::new();
    assert!(apply_resolved_auth(
        &mut headers,
        &auth,
        Some(now),
        "https://downloads.example/file",
        REQUEST_KIND_ARTIFACT,
        now
    )
    .unwrap_err()
    .to_string()
    .contains("expired"));
    assert!(request_url_allowed(&UrlPolicy::default(), "http://127.0.0.1:1234/file").is_err());
    assert!(request_url_allowed(
        &UrlPolicy::default().allow_http_origin("http://127.0.0.1:1234"),
        "http://127.0.0.1:1234/file"
    )
    .is_ok());
}

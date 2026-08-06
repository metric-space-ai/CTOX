// ref: sdk/cliproxy/auth/classification_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use serde_json::json;

use super::{Auth, AuthKind, AuthSourceKind};

#[test]
fn auth_kind_matches_explicit_and_legacy_precedence() {
    let cases = [
        (
            {
                let mut auth = Auth::default();
                auth.attributes.insert("auth_kind".into(), "api_key".into());
                auth
            },
            Some(AuthKind::ApiKey),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.attributes.insert("auth_kind".into(), "oauth".into());
                auth.attributes.insert("api_key".into(), "k".into());
                auth
            },
            Some(AuthKind::OAuth),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.metadata.insert("auth_kind".into(), json!("oauth"));
                auth
            },
            Some(AuthKind::OAuth),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.attributes.insert("api_key".into(), "k".into());
                auth
            },
            Some(AuthKind::ApiKey),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.metadata.insert("access_token".into(), json!("token"));
                auth
            },
            Some(AuthKind::OAuth),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.metadata.insert("type".into(), json!("test"));
                auth
            },
            None,
        ),
    ];
    for (auth, expected) in cases {
        assert_eq!(auth.auth_kind(), expected);
    }
}

#[test]
fn auth_source_kind_matches_runtime_backend_and_file_fallbacks() {
    let cases = [
        (
            {
                let mut auth = Auth::default();
                auth.attributes.insert("runtime_only".into(), "true".into());
                auth.attributes
                    .insert("source_backend".into(), "postgres".into());
                auth
            },
            Some(AuthSourceKind::Memory),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.attributes
                    .insert("source_backend".into(), "postgresql".into());
                auth.attributes
                    .insert("path".into(), "/tmp/auth.json".into());
                auth
            },
            Some(AuthSourceKind::Postgres),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.attributes
                    .insert("source_backend".into(), "object-store".into());
                auth
            },
            Some(AuthSourceKind::ObjectStore),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.attributes
                    .insert("source".into(), "config:codex[abc]".into());
                auth
            },
            Some(AuthSourceKind::Config),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.attributes
                    .insert("source".into(), "/tmp/auth.json".into());
                auth
            },
            Some(AuthSourceKind::File),
        ),
        (
            {
                let mut auth = Auth::default();
                auth.file_name = "codex.json".into();
                auth
            },
            Some(AuthSourceKind::File),
        ),
    ];
    for (auth, expected) in cases {
        assert_eq!(auth.auth_source_kind(), expected);
    }
}

#[test]
fn account_info_uses_classified_kind() {
    let mut api_key = Auth::default();
    api_key
        .attributes
        .insert("auth_kind".into(), "api-key".into());
    api_key.attributes.insert("api_key".into(), "k".into());
    assert_eq!(api_key.account_info(), ("api_key".into(), "k".into()));

    let mut oauth = Auth::default();
    oauth.attributes.insert("auth_kind".into(), "oauth".into());
    oauth.attributes.insert("api_key".into(), "k".into());
    oauth
        .metadata
        .insert("email".into(), json!(" user@example.com "));
    assert_eq!(
        oauth.account_info(),
        ("oauth".into(), "user@example.com".into())
    );
}

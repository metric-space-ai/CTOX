// ref: sdk/cliproxy/auth/conductor_claude_cancellation_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: typed request-scoped errors replace Go context cancellation inspection
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use crate::sdk::pluginapi::PluginExecutionError;

use super::{
    is_claude_oauth_request_cancellation, is_request_scoped_plugin_error, Auth, AuthError,
};

fn oauth(provider: &str) -> Auth {
    let mut auth = Auth::default();
    auth.provider = provider.to_owned();
    auth.metadata
        .insert("access_token".into(), serde_json::json!("token"));
    auth.metadata
        .insert("refresh_token".into(), serde_json::json!("refresh"));
    auth
}

fn cancellation() -> PluginExecutionError {
    Arc::new(AuthError {
        code: "request_scoped".into(),
        message: "request cancelled".into(),
        ..AuthError::default()
    })
}

#[test]
fn claude_oauth_request_cancellation_is_direct_and_availability_neutral() {
    let error = cancellation();
    assert!(is_request_scoped_plugin_error(&error));
    assert!(is_claude_oauth_request_cancellation(
        &oauth("claude"),
        &error
    ));
}

#[test]
fn cancellation_classifier_does_not_change_other_provider_or_auth_kinds() {
    let error = cancellation();
    assert!(!is_claude_oauth_request_cancellation(
        &oauth("codex"),
        &error
    ));
    let mut api_key = Auth::default();
    api_key.provider = "claude".into();
    api_key.attributes.insert("api_key".into(), "key".into());
    assert!(!is_claude_oauth_request_cancellation(&api_key, &error));
}

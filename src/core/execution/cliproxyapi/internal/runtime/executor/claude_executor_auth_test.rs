// ref: internal/runtime/executor/claude_executor_auth_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{
    prepare_claude_request_auth, should_prepare_claude_request_auth, ClaudeOAuthProfile,
    ClaudeOAuthProfileFetcher, ClaudeRequestAuthPreparer, CLAUDE_ACCOUNT_PROFILE_CHECKED_AT_KEY,
};
use crate::sdk::cliproxy::auth::AuthPreparer;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

struct Profile;
impl ClaudeOAuthProfileFetcher for Profile {
    fn fetch<'a>(
        &'a self,
        _: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<ClaudeOAuthProfile, String>> + Send + 'a>> {
        Box::pin(async {
            Ok(ClaudeOAuthProfile {
                account_uuid: "account-a".into(),
                email: "a@example.test".into(),
                organization_uuid: "org-a".into(),
                organization_name: "Org".into(),
            })
        })
    }
}

#[tokio::test]
async fn oauth_request_auth_publishes_device_and_account_identity() {
    let mut auth = crate::sdk::cliproxy::auth::Auth::default();
    auth.id = "claude-auth-test".into();
    auth.attributes
        .insert("api_key".into(), "sk-ant-oat-test".into());
    prepare_claude_request_auth(&mut auth, None, &Profile)
        .await
        .unwrap();
    assert_eq!(
        auth.metadata
            .get("account_uuid")
            .and_then(serde_json::Value::as_str),
        Some("account-a")
    );
    assert!(auth
        .metadata
        .get(CLAUDE_ACCOUNT_PROFILE_CHECKED_AT_KEY)
        .and_then(serde_json::Value::as_str)
        .is_some_and(|value| value.ends_with('Z')));
    assert!(!should_prepare_claude_request_auth(&auth));
}

#[tokio::test]
async fn oauth_request_auth_accepts_selected_metadata_token() {
    let mut auth = crate::sdk::cliproxy::auth::Auth::default();
    auth.id = "claude-metadata-token".into();
    auth.metadata.insert(
        "access_token".into(),
        serde_json::json!("sk-ant-oat-metadata"),
    );
    assert!(should_prepare_claude_request_auth(&auth));
    prepare_claude_request_auth(&mut auth, None, &Profile)
        .await
        .unwrap();
    assert_eq!(
        auth.metadata
            .get("organization_uuid")
            .and_then(serde_json::Value::as_str),
        Some("org-a")
    );
}

#[tokio::test]
async fn conductor_capability_delegates_to_claude_auth_preparation() {
    let preparer = ClaudeRequestAuthPreparer::new(None, Arc::new(Profile));
    let mut auth = crate::sdk::cliproxy::auth::Auth::default();
    auth.id = "claude-capability".into();
    auth.attributes
        .insert("api_key".into(), "sk-ant-oat-capability".into());

    preparer.prepare(&mut auth).await.unwrap();

    assert_eq!(
        auth.metadata
            .get("account_uuid")
            .and_then(serde_json::Value::as_str),
        Some("account-a")
    );
    assert!(!should_prepare_claude_request_auth(&auth));
}

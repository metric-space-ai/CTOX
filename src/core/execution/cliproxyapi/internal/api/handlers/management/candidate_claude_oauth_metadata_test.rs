// ref: internal/api/handlers/management/auth_files_provider_oauth.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: AGPL-3.0-only

use std::time::{Duration, SystemTime};

use crate::internal::auth::claude::{
    ClaudeTokenData, ClaudeTokenStorage, SecretString, CLAUDE_DEVICE_IDS_METADATA_KEY,
};

use super::claude_oauth_runtime_metadata;

#[test]
fn claude_oauth_metadata_preserves_identity_without_token_material() {
    let data = ClaudeTokenData::new(
        SecretString::new("access-secret").unwrap(),
        SecretString::new("refresh-secret").unwrap(),
        " user@example.com ",
        SystemTime::UNIX_EPOCH + Duration::from_secs(3_600),
    )
    .with_identity("account-1", "org-1", "Example Org");
    let storage = ClaudeTokenStorage::from_token_data(&data, SystemTime::UNIX_EPOCH, None)
        .with_device_ids(&["a".repeat(64)]);
    let metadata = claude_oauth_runtime_metadata(&storage);
    // Candidate stores tokenStorage.Email verbatim; display-time trimming is a
    // separate management concern and must not mutate credential identity.
    assert_eq!(metadata["email"], " user@example.com ");
    assert_eq!(metadata["account_uuid"], "account-1");
    assert_eq!(metadata["organization_uuid"], "org-1");
    assert_eq!(metadata["organization_name"], "Example Org");
    assert_eq!(
        metadata[CLAUDE_DEVICE_IDS_METADATA_KEY],
        serde_json::json!(["a".repeat(64)])
    );
    let encoded = serde_json::to_string(&metadata).unwrap();
    assert!(!encoded.contains("access-secret"));
    assert!(!encoded.contains("refresh-secret"));
}

#[test]
fn empty_optional_identity_fields_are_omitted() {
    let data = ClaudeTokenData::new(
        SecretString::new("access").unwrap(),
        SecretString::new("refresh").unwrap(),
        "email@example.com",
        SystemTime::UNIX_EPOCH,
    );
    let storage = ClaudeTokenStorage::from_token_data(&data, SystemTime::UNIX_EPOCH, None);
    let metadata = claude_oauth_runtime_metadata(&storage);
    assert_eq!(metadata.len(), 1);
    assert_eq!(metadata["email"], "email@example.com");
}

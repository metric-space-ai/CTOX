// ref: internal/runtime/executor/helps/derived_session_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use uuid::Uuid;

use super::{derived_antigravity_session_id, derived_session_uuid, provider_session_uuid};
use crate::sdk::cliproxy::executor::ExecutionMetadata;

#[test]
fn derived_session_provider_mappings() {
    let metadata = ExecutionMetadata {
        derived_session_id: Some("ctx:v1:test-root".into()),
        ..ExecutionMetadata::default()
    };
    let codex_id = derived_session_uuid("codex", &[&metadata]);
    let xai_id = derived_session_uuid("xai", &[&metadata]);
    assert!(Uuid::parse_str(&codex_id).is_ok());
    assert!(Uuid::parse_str(&xai_id).is_ok());
    assert_ne!(codex_id, xai_id);
    assert_eq!(derived_session_uuid("codex", &[&metadata]), codex_id);

    let antigravity_id = derived_antigravity_session_id(&[&metadata]);
    assert!(antigravity_id
        .strip_prefix('-')
        .is_some_and(|value| !value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit())));
    assert_eq!(derived_antigravity_session_id(&[&metadata]), antigravity_id);
}

#[test]
fn provider_session_uuid_prefers_execution_session() {
    let first = ExecutionMetadata {
        execution_session_id: Some("connection-1".into()),
        derived_session_id: Some("ctx:v1:first-root".into()),
        ..ExecutionMetadata::default()
    };
    let second = ExecutionMetadata {
        execution_session_id: Some("connection-1".into()),
        derived_session_id: Some("ctx:v1:second-root".into()),
        ..ExecutionMetadata::default()
    };
    let first_id = provider_session_uuid("codex", &[&first]);
    let second_id = provider_session_uuid("codex", &[&second]);
    assert!(!first_id.is_empty());
    assert_eq!(first_id, second_id);
    assert_ne!(first_id, derived_session_uuid("codex", &[&first]));
}

#[test]
fn derived_session_provider_mappings_require_identity() {
    assert!(derived_session_uuid("codex", &[]).is_empty());
    assert!(derived_antigravity_session_id(&[]).is_empty());
}

#[test]
fn metadata_precedence_and_normalization_match_upstream() {
    let empty = ExecutionMetadata {
        derived_session_id: Some("  ".into()),
        execution_session_id: Some("\t".into()),
        ..ExecutionMetadata::default()
    };
    let first = ExecutionMetadata {
        derived_session_id: Some(" ctx:v1:first ".into()),
        execution_session_id: Some(" execution-first ".into()),
        ..ExecutionMetadata::default()
    };
    let later = ExecutionMetadata {
        derived_session_id: Some("ctx:v1:later".into()),
        execution_session_id: Some("execution-later".into()),
        ..ExecutionMetadata::default()
    };
    assert_eq!(
        super::derived_session_id(&[&empty, &first, &later]),
        "ctx:v1:first"
    );
    assert_eq!(
        provider_session_uuid(" CODEX ", &[&empty, &first, &later]),
        provider_session_uuid("codex", &[&first])
    );
    assert!(derived_session_uuid("", &[&first]).is_empty());
}

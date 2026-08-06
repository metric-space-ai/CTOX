// ref: internal/auth/codex/filename_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::token::{CodexCredentialHandles, CodexSecretHandle, CodexSecretKind};

fn handles(account_id: &str) -> CodexCredentialHandles {
    let handle = |suffix, kind| {
        CodexSecretHandle::new(
            "subscriptions",
            format!("codex-{account_id}-{suffix}"),
            kind,
        )
        .unwrap()
    };
    CodexCredentialHandles::new(
        handle("id", CodexSecretKind::IdToken),
        handle("access", CodexSecretKind::AccessToken),
        handle("refresh", CodexSecretKind::RefreshToken),
    )
    .unwrap()
}

#[test]
fn account_identity_keeps_same_email_and_plan_credentials_distinct_without_files() {
    let first = handles("abc12345");
    let second = handles("def67890");

    assert_ne!(first, second);
    assert_eq!(first.id_token().scope(), "subscriptions");
    assert_eq!(first.id_token().name(), "codex-abc12345-id");
    assert_eq!(second.id_token().name(), "codex-def67890-id");
    let debug = format!("{first:?}{second:?}");
    assert!(!debug.contains("user@example.com"));
    assert!(!debug.contains(".json"));
    assert!(!debug.contains('/'));
}

#[test]
fn one_account_snapshot_uses_three_distinct_typed_records() {
    let handles = handles("abc12345");
    assert_eq!(handles.id_token().kind(), CodexSecretKind::IdToken);
    assert_eq!(handles.access_token().kind(), CodexSecretKind::AccessToken);
    assert_eq!(
        handles.refresh_token().kind(),
        CodexSecretKind::RefreshToken
    );
    assert_ne!(handles.id_token().name(), handles.access_token().name());
    assert_ne!(handles.id_token().name(), handles.refresh_token().name());
    assert_ne!(
        handles.access_token().name(),
        handles.refresh_token().name()
    );
}

// ref: sdk/cliproxy/auth/selected_auth_metadata_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use crate::sdk::cliproxy::executor::ExecutionMetadata;

use super::{publish_selected_auth_metadata, Auth};

#[test]
fn selected_auth_metadata_includes_stable_index() {
    let mut auth = Auth::default();
    auth.id = "auth-1".to_owned();
    auth.provider = "codex".to_owned();
    auth.file_name = "auth-1.json".to_owned();

    let selected_id = Arc::new(Mutex::new(String::new()));
    let selected_index = Arc::new(Mutex::new(String::new()));
    let mut metadata = ExecutionMetadata {
        selected_auth_callback: Some({
            let selected_id = selected_id.clone();
            Arc::new(move |value| {
                *selected_id
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = value.to_owned();
            })
        }),
        selected_auth_index_callback: Some({
            let selected_index = selected_index.clone();
            Arc::new(move |value| {
                *selected_index
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner) = value.to_owned();
            })
        }),
        ..ExecutionMetadata::default()
    };
    publish_selected_auth_metadata(&mut metadata, &mut auth);

    assert_eq!(metadata.selected_auth_id.as_deref(), Some("auth-1"));
    assert_eq!(
        metadata.selected_auth_index.as_deref(),
        Some(auth.index.as_str())
    );
    assert!(!auth.index.is_empty());
    assert_eq!(
        selected_id
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_str(),
        "auth-1"
    );
    assert_eq!(
        selected_index
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .as_str(),
        auth.index
    );
}

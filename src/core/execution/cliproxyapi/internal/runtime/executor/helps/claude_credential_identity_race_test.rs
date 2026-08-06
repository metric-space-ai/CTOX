// ref: internal/runtime/executor/helps/claude_credential_identity_race_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use serde_json::json;

use super::claude_credential_identity::{
    apply_claude_credential_metadata, ensure_claude_credential_device_pool_required,
};
use crate::internal::auth::claude::CLAUDE_DEVICE_IDS_METADATA_KEY;
use crate::sdk::cliproxy::auth::Auth;

#[test]
fn concurrent_shared_auth_has_one_stable_device_pool() {
    let mut shared_auth = Auth::default();
    shared_auth.id = "shared-credential".to_owned();
    shared_auth.metadata.insert(
        "account_uuid".to_owned(),
        json!("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
    );
    let auth = Arc::new(Mutex::new(shared_auth));
    let mut workers = Vec::new();
    for index in 0..32 {
        let auth = Arc::clone(&auth);
        workers.push(std::thread::spawn(move || {
            let mut auth = auth.lock().unwrap_or_else(|error| error.into_inner());
            let session = format!("session-{}", char::from(b'a' + index % 26));
            let (_, device) = apply_claude_credential_metadata(
                br#"{"model":"claude-opus-4-6","messages":[]}"#,
                &mut auth,
                &session,
            )?;
            ensure_claude_credential_device_pool_required(None, &mut auth)?;
            Ok::<_, super::claude_credential_identity::ClaudeCredentialIdentityError>(device)
        }));
    }
    let devices = workers
        .into_iter()
        .map(|worker| worker.join().unwrap().unwrap())
        .collect::<std::collections::BTreeSet<_>>();
    assert_eq!(devices.len(), 1);
    let auth = auth.lock().unwrap();
    assert!(auth.metadata.contains_key(CLAUDE_DEVICE_IDS_METADATA_KEY));
}

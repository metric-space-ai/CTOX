// candidate-evidence: sdk/cliproxy/auth/token_fingerprint.go
// @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: supplemental
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use serde_json::json;

use super::{access_token_sha256, notify_access_token_fingerprint, Auth};

fn auth_with_token(metadata: serde_json::Value) -> Auth {
    let mut auth = Auth::default();
    auth.id = "home-auth".to_owned();
    auth.provider = " claude ".to_owned();
    auth.metadata = metadata
        .as_object()
        .expect("fixture metadata object")
        .iter()
        .map(|(key, value)| (key.clone(), value.clone()))
        .collect();
    auth
}

#[test]
fn candidate_token_fingerprint_matches_go_precedence_and_normalization() {
    let direct = auth_with_token(json!({
        "access_token": "  direct-token  ",
        "token": {"accessToken": "nested-token"}
    }));
    assert_eq!(
        access_token_sha256(&direct),
        "e38fd6b13248707d5c1530008912c2aeef94d7aa64d251eef7c7351b7fb7cc1d"
    );
    let nested = auth_with_token(json!({"Token": {"accessToken": " nested-token "}}));
    assert_eq!(
        access_token_sha256(&nested),
        "f7fa9cf7f8b000562e1db5a78bd04de4d1aa97eb8b0fd86b91f2ac80e063defb"
    );
    assert!(access_token_sha256(&Auth::default()).is_empty());
}

#[test]
fn candidate_observer_receives_owned_snapshot_without_debugging_token() {
    let observed = Arc::new(Mutex::new(None));
    let sink = Arc::clone(&observed);
    let observer = move |observation: super::AccessTokenFingerprintObservation| {
        *sink.lock().unwrap() = Some(observation);
    };
    let auth = auth_with_token(json!({"access_token": "secret-token"}));
    assert!(notify_access_token_fingerprint(Some(&observer), &auth));
    let observation = observed.lock().unwrap().take().unwrap();
    let rendered = format!("{observation:?}");
    assert!(!rendered.contains("secret-token"));
    assert_eq!(observation.fingerprint(), access_token_sha256(&auth));
    assert_eq!(
        observation.into_auth().metadata["access_token"],
        "secret-token"
    );
    assert!(!notify_access_token_fingerprint(None, &auth));
}

// ref: internal/runtime/executor/helps/claude_credential_identity_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::Mutex;

use serde_json::{json, Value};

use super::claude_credential_identity::{
    apply_claude_credential_metadata, claude_agent_session_uuid_for_request,
    ensure_claude_credential_device_pool_required, ClaudeCredentialDevicePoolStore,
    ClaudeCredentialIdentityError,
};
use crate::internal::auth::claude::CLAUDE_DEVICE_IDS_METADATA_KEY;
use crate::internal::home::hash_key_part;
use crate::sdk::api::handlers::header_filter::HeaderMap;
use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::cliproxy::executor::ExecutionMetadata;

#[derive(Default)]
struct FakePoolStore {
    values: Mutex<BTreeMap<String, Vec<u8>>>,
    operations: Mutex<Vec<&'static str>>,
}

impl ClaudeCredentialDevicePoolStore for FakePoolStore {
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, ClaudeCredentialIdentityError> {
        Ok(self.values.lock().unwrap().get(key).cloned())
    }

    fn set_nx(&self, key: &str, value: &[u8]) -> Result<bool, ClaudeCredentialIdentityError> {
        let mut values = self.values.lock().unwrap();
        if values.contains_key(key) {
            return Ok(false);
        }
        values.insert(key.to_owned(), value.to_vec());
        self.operations.lock().unwrap().push("nx");
        Ok(true)
    }

    fn set_existing(&self, key: &str, value: &[u8]) -> Result<bool, ClaudeCredentialIdentityError> {
        let mut values = self.values.lock().unwrap();
        if !values.contains_key(key) {
            return Ok(false);
        }
        values.insert(key.to_owned(), value.to_vec());
        self.operations.lock().unwrap().push("xx");
        Ok(true)
    }
}

#[test]
fn preserves_confirmed_native_session_and_ignores_unconfirmed_claude_signals() {
    let native = "11111111-2222-4333-8444-555555555555";
    let headers = HeaderMap::from([(
        "X-Claude-Code-Session-Id".to_owned(),
        vec![native.to_owned()],
    )]);
    assert_eq!(
        claude_agent_session_uuid_for_request(Some(&headers), &[], &[], true, &[]),
        native
    );
    let metadata = ExecutionMetadata {
        execution_session_id: Some("non-native-conversation".to_owned()),
        ..ExecutionMetadata::default()
    };
    let first = claude_agent_session_uuid_for_request(
        Some(&headers),
        br#"{"metadata":{"user_id":"{\"device_id\":\"0000000000000000000000000000000000000000000000000000000000000000\",\"session_id\":\"11111111-2222-4333-8444-555555555555\"}"}}"#,
        &[],
        false,
        &[&metadata],
    );
    let repeated = claude_agent_session_uuid_for_request(None, &[], &[], false, &[&metadata]);
    assert_ne!(first, native);
    assert_eq!(first, repeated);
}

#[test]
fn migrates_home_pool_to_one_canonical_device() {
    let legacy = (0..5)
        .map(|digit| digit.to_string().repeat(64))
        .collect::<Vec<_>>();
    let store = FakePoolStore::default();
    let key = format!(
        "cpa:claude:credential-device-pool:{}",
        hash_key_part("legacy-five-device-credential")
    );
    store
        .values
        .lock()
        .unwrap()
        .insert(key, serde_json::to_vec(&legacy).unwrap());
    let mut auth = Auth::default();
    auth.id = "legacy-five-device-credential".to_owned();
    auth.index = "legacy-five-device-credential".to_owned();
    let pool = ensure_claude_credential_device_pool_required(Some(&store), &mut auth).unwrap();
    assert_eq!(pool, vec![legacy[0].clone()]);
    assert_eq!(*store.operations.lock().unwrap(), vec!["xx"]);
    assert_eq!(
        auth.metadata[CLAUDE_DEVICE_IDS_METADATA_KEY],
        json!([legacy[0]])
    );
}

#[test]
fn applies_credential_identity_and_preserves_encoded_extras() {
    let device = "0".repeat(64);
    let account = "aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa";
    let session = "11111111-2222-4333-8444-555555555555";
    let mut auth = Auth::default();
    auth.metadata
        .insert("account_uuid".to_owned(), json!(account));
    auth.metadata
        .insert(CLAUDE_DEVICE_IDS_METADATA_KEY.to_owned(), json!([device]));
    let existing = json!({
        "device_id": "f".repeat(64),
        "account_uuid": "downstream",
        "session_id": "downstream",
        "parent_session_id": "parent-1",
        "extra": true,
    });
    let body = serde_json::to_vec(&json!({
        "messages": [],
        "metadata": {"user_id": serde_json::to_string(&existing).unwrap()}
    }))
    .unwrap();
    let (updated, selected) = apply_claude_credential_metadata(&body, &mut auth, session).unwrap();
    let root: Value = serde_json::from_slice(&updated).unwrap();
    let encoded = root["metadata"]["user_id"].as_str().unwrap();
    let identity: Value = serde_json::from_str(encoded).unwrap();
    assert_eq!(identity["device_id"], selected);
    assert_eq!(identity["account_uuid"], account);
    assert_eq!(identity["session_id"], session);
    assert_eq!(identity["parent_session_id"], "parent-1");
    assert_eq!(identity["extra"], true);
    assert!(encoded.starts_with(&format!("{{\"device_id\":\"{selected}\",\"account_uuid\":")));
}

#[test]
fn rejects_duplicate_identity_containers_as_request_scoped_400() {
    let mut auth = Auth::default();
    auth.metadata.insert(
        "account_uuid".to_owned(),
        json!("aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"),
    );
    auth.metadata.insert(
        CLAUDE_DEVICE_IDS_METADATA_KEY.to_owned(),
        json!(["0".repeat(64)]),
    );
    for body in [
        br#"{"messages":[],"metadata":{"user_id":"{}"},"metadata":{"user_id":"{}"}}"#.as_slice(),
        br#"{"messages":[],"metadata":{"user_id":"{}","user_id":"{}"}}"#.as_slice(),
        br#"{"messages":[],"metadata":{"user_id":"{\"account_uuid\":\"first\",\"account_uuid\":\"last\"}"}}"#.as_slice(),
    ] {
        let error = apply_claude_credential_metadata(
            body,
            &mut auth,
            "11111111-2222-4333-8444-555555555555",
        )
        .unwrap_err();
        assert!(error.is_request_scoped());
        assert_eq!(error.status_code(), Some(400));
    }
}

#[test]
fn missing_account_uuid_is_credential_scoped() {
    let mut auth = Auth::default();
    auth.metadata.insert(
        CLAUDE_DEVICE_IDS_METADATA_KEY.to_owned(),
        json!(["0".repeat(64)]),
    );
    let error = apply_claude_credential_metadata(
        br#"{"messages":[]}"#,
        &mut auth,
        "11111111-2222-4333-8444-555555555555",
    )
    .unwrap_err();
    assert!(!error.is_request_scoped());
}

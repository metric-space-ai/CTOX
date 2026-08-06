// ref: internal/runtime/executor/helps/home_refresh_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};

use super::home_refresh::*;
use crate::sdk::cliproxy::auth::Auth;

struct FakeClient {
    heartbeat: bool,
    response: Mutex<Result<Vec<u8>, HomeRefreshClientError>>,
    calls: Mutex<Vec<(String, String)>>,
}

impl FakeClient {
    fn responding(value: Value) -> Self {
        Self {
            heartbeat: true,
            response: Mutex::new(Ok(serde_json::to_vec(&value).unwrap())),
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl HomeRefreshClient for FakeClient {
    fn heartbeat_ok(&self) -> bool {
        self.heartbeat
    }

    fn get_refresh_auth<'a>(
        &'a self,
        auth_index: &'a str,
        access_token_sha256: &'a str,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<u8>, HomeRefreshClientError>> + Send + 'a>> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap()
                .push((auth_index.to_owned(), access_token_sha256.to_owned()));
            self.response.lock().unwrap().clone()
        })
    }
}

fn auth_with_index(index: &str) -> Auth {
    let mut auth = Auth::default();
    auth.id = "home-auth-1".to_owned();
    auth.index = index.to_owned();
    auth.provider = "antigravity".to_owned();
    auth.metadata = BTreeMap::from([
        (
            "refresh_token".to_owned(),
            Value::String("refresh-token".to_owned()),
        ),
        (
            "access_token".to_owned(),
            Value::String("old-access-token".to_owned()),
        ),
    ]);
    auth
}

#[test]
fn status_mapping_matches_upstream() {
    assert_eq!(status_from_home_error_code("authentication_error"), 401);
    assert_eq!(status_from_home_error_code(" unauthorized "), 401);
    assert_eq!(status_from_home_error_code("MODEL_NOT_FOUND"), 404);
    assert_eq!(status_from_home_error_code("invalid_grant"), 401);
    assert_eq!(status_from_home_error_code("auth_unavailable"), 503);
    assert_eq!(status_from_home_error_code("anything_else"), 503);
}

#[tokio::test]
async fn accepts_auth_envelope() {
    let client = Arc::new(FakeClient::responding(json!({
        "auth": {
            "id": "home-auth-1",
            "provider": "antigravity",
            "metadata": {"access_token": "new-access-token"}
        },
        "auth_index": "home-index-1"
    })));
    let authority = HomeRefreshAuthority::enabled(client.clone());
    let updated = match authority
        .refresh_auth_via_home(Some(&auth_with_index("home-index-1")))
        .await
        .unwrap()
    {
        HomeRefreshDisposition::Refreshed(auth) => auth,
        HomeRefreshDisposition::Disabled => panic!("enabled authority returned disabled"),
    };
    let calls = client.calls.lock().unwrap();
    assert_eq!(calls[0].0, "home-index-1");
    assert_eq!(calls[0].1.len(), 64);
    drop(calls);
    assert_eq!(
        updated.metadata.get("access_token"),
        Some(&Value::String("new-access-token".to_owned()))
    );
    assert_eq!(updated.index, "home-index-1");
}

#[tokio::test]
async fn disabled_mode_is_not_handled_and_never_requires_auth() {
    assert!(matches!(
        HomeRefreshAuthority::disabled()
            .refresh_auth_via_home(None)
            .await
            .unwrap(),
        HomeRefreshDisposition::Disabled
    ));
}

#[tokio::test]
async fn enabled_mode_validates_auth_and_heartbeat_before_transport() {
    let client = Arc::new(FakeClient {
        heartbeat: false,
        response: Mutex::new(Ok(Vec::new())),
        calls: Mutex::new(Vec::new()),
    });
    let authority = HomeRefreshAuthority::enabled(client.clone());
    let nil_error = authority.refresh_auth_via_home(None).await.unwrap_err();
    assert_eq!(nil_error.status_code(), 500);
    let unavailable = authority
        .refresh_auth_via_home(Some(&auth_with_index("index")))
        .await
        .unwrap_err();
    assert_eq!(unavailable.status_code(), 503);
    assert!(client.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn derives_missing_index_and_accepts_direct_auth_payload() {
    let client = Arc::new(FakeClient::responding(json!({
        "id": "refreshed",
        "provider": "claude",
        "metadata": {"access_token": "new"}
    })));
    let authority = HomeRefreshAuthority::enabled(client.clone());
    let mut original = Auth::default();
    original.id = "stable-id".to_owned();
    original.provider = "claude".to_owned();
    let updated = match authority
        .refresh_auth_via_home(Some(&original))
        .await
        .unwrap()
    {
        HomeRefreshDisposition::Refreshed(auth) => auth,
        HomeRefreshDisposition::Disabled => unreachable!(),
    };
    let called_index = client.calls.lock().unwrap()[0].0.clone();
    assert!(!called_index.is_empty());
    assert_eq!(updated.index, called_index);
    assert_eq!(updated.id, "refreshed");
}

#[tokio::test]
async fn empty_unindexable_auth_is_a_bad_gateway() {
    let client = Arc::new(FakeClient::responding(json!({})));
    let error = HomeRefreshAuthority::enabled(client.clone())
        .refresh_auth_via_home(Some(&Auth::default()))
        .await
        .unwrap_err();
    assert_eq!(error.status_code(), 502);
    assert!(client.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn maps_home_error_envelopes_and_defaults_blank_messages() {
    for (code, expected_status) in [
        ("authentication_error", 401),
        ("model_not_found", 404),
        ("rate_limited", 503),
    ] {
        let client = Arc::new(FakeClient::responding(json!({
            "error": {"type": code, "message": ""}
        })));
        let error = HomeRefreshAuthority::enabled(client)
            .refresh_auth_via_home(Some(&auth_with_index("index")))
            .await
            .unwrap_err();
        assert_eq!(error.status_code(), expected_status);
        assert_eq!(
            error.to_string(),
            match expected_status {
                401 => "credential unauthorized",
                404 => "credential refresh target not found",
                _ => "credential refresh temporarily unavailable",
            }
        );
    }
}

#[tokio::test]
async fn transport_invalid_json_and_oversized_payload_fail_closed() {
    let transport_error = Arc::new(FakeClient {
        heartbeat: true,
        response: Mutex::new(Err(HomeRefreshClientError::new("transport failed"))),
        calls: Mutex::new(Vec::new()),
    });
    let error = HomeRefreshAuthority::enabled(transport_error)
        .refresh_auth_via_home(Some(&auth_with_index("index")))
        .await
        .unwrap_err();
    assert_eq!(error.status_code(), 503);
    assert_eq!(error.to_string(), "home refresh temporarily unavailable");

    for raw in [
        b"not-json".to_vec(),
        vec![b' '; MAX_HOME_REFRESH_PAYLOAD_BYTES + 1],
    ] {
        let client = Arc::new(FakeClient {
            heartbeat: true,
            response: Mutex::new(Ok(raw)),
            calls: Mutex::new(Vec::new()),
        });
        let error = HomeRefreshAuthority::enabled(client)
            .refresh_auth_via_home(Some(&auth_with_index("index")))
            .await
            .unwrap_err();
        assert_eq!(error.status_code(), 502);
        assert_eq!(error.to_string(), "home returned invalid auth payload");
    }
}

#[tokio::test]
async fn preserves_cancellation_and_deadline_classification() {
    for (error, expected) in [
        (
            HomeRefreshClientError::cancelled("cancelled"),
            HomeRefreshClientErrorKind::Cancelled,
        ),
        (
            HomeRefreshClientError::deadline_exceeded("deadline"),
            HomeRefreshClientErrorKind::DeadlineExceeded,
        ),
    ] {
        let client = Arc::new(FakeClient {
            heartbeat: true,
            response: Mutex::new(Err(error)),
            calls: Mutex::new(Vec::new()),
        });
        let error = HomeRefreshAuthority::enabled(client)
            .refresh_auth_via_home(Some(&auth_with_index("index")))
            .await
            .unwrap_err();
        assert_eq!(error.client_error_kind(), expected);
        assert_eq!(error.status_code(), 0);
    }
}

#[tokio::test]
async fn rejects_disabled_refreshed_credentials() {
    let client = Arc::new(FakeClient::responding(json!({
        "id": "disabled",
        "provider": "claude",
        "disabled": true
    })));
    let error = HomeRefreshAuthority::enabled(client)
        .refresh_auth_via_home(Some(&auth_with_index("index")))
        .await
        .unwrap_err();
    assert_eq!(error.status_code(), 401);
    assert_eq!(error.to_string(), "credential unauthorized");
}

#[tokio::test]
async fn injected_authorities_are_isolated_and_debug_redacts_client() {
    let first = Arc::new(FakeClient::responding(json!({
        "id": "first", "provider": "claude"
    })));
    let second = Arc::new(FakeClient::responding(json!({
        "id": "second", "provider": "claude"
    })));
    let first_authority = HomeRefreshAuthority::enabled(first);
    let second_authority = HomeRefreshAuthority::enabled(second);
    let first_id = match first_authority
        .refresh_auth_via_home(Some(&auth_with_index("one")))
        .await
        .unwrap()
    {
        HomeRefreshDisposition::Refreshed(auth) => auth.id,
        HomeRefreshDisposition::Disabled => unreachable!(),
    };
    let second_id = match second_authority
        .refresh_auth_via_home(Some(&auth_with_index("two")))
        .await
        .unwrap()
    {
        HomeRefreshDisposition::Refreshed(auth) => auth.id,
        HomeRefreshDisposition::Disabled => unreachable!(),
    };
    assert_eq!((first_id.as_str(), second_id.as_str()), ("first", "second"));
    assert_eq!(
        format!("{first_authority:?}"),
        "HomeRefreshAuthority { enabled: true, .. }"
    );
}

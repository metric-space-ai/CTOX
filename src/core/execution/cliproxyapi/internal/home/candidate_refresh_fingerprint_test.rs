// ref: internal/home/requests.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: AGPL-3.0-only

use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::client::{
    Client, HomeConfig, HomeError, HomeTlsConfig, HomeTransport, KvSetOptions, TransportFailure,
    HOME_REFRESH_OPERATION_TIMEOUT,
};

#[derive(Default)]
struct ProbeTransport {
    observation: Mutex<Option<(String, Vec<u8>, Duration)>>,
}

impl HomeTransport for ProbeTransport {
    fn ping(&self) -> Result<(), HomeError> {
        Ok(())
    }
    fn get(&self, _: &str) -> Result<Option<Vec<u8>>, HomeError> {
        Ok(None)
    }
    fn set(&self, _: &str, _: &[u8], _: KvSetOptions) -> Result<bool, HomeError> {
        Ok(true)
    }
    fn compare_and_swap(
        &self,
        _: &str,
        _: Option<&[u8]>,
        _: &[u8],
        _: Duration,
    ) -> Result<bool, HomeError> {
        Ok(true)
    }
    fn delete(&self, _: &[String]) -> Result<i64, HomeError> {
        Ok(0)
    }
    fn expire(&self, _: &str, _: Duration) -> Result<bool, HomeError> {
        Ok(true)
    }
    fn ttl(&self, _: &str) -> Result<Option<Duration>, HomeError> {
        Ok(None)
    }
    fn increment(&self, _: &str, _: i64) -> Result<i64, HomeError> {
        Ok(0)
    }
    fn push(&self, _: &str, _: &[u8], _: bool) -> Result<(), HomeError> {
        Ok(())
    }
    fn request(&self, _: &str, _: &[u8]) -> Result<Vec<u8>, TransportFailure> {
        panic!("refresh must use request_with_timeout")
    }
    fn request_with_timeout(
        &self,
        key: &str,
        payload: &[u8],
        timeout: Duration,
    ) -> Result<Vec<u8>, TransportFailure> {
        *self
            .observation
            .lock()
            .unwrap_or_else(|error| error.into_inner()) =
            Some((key.to_owned(), payload.to_vec(), timeout));
        Ok(br#"{"id":"auth"}"#.to_vec())
    }
}

#[test]
fn refresh_carries_observed_fingerprint_and_uses_extended_timeout() {
    let transport = Arc::new(ProbeTransport::default());
    let facade: Arc<dyn HomeTransport> = transport.clone();
    let client = Client::new(
        HomeConfig {
            enabled: true,
            host: "home".into(),
            port: 6379,
            tls: HomeTlsConfig::default(),
            ..HomeConfig::default()
        },
        facade,
    );
    client
        .get_refresh_auth_with_fingerprint(" auth-1 ", " abc123 ")
        .expect("refresh response");
    let (key, payload, timeout) = transport
        .observation
        .lock()
        .unwrap_or_else(|error| error.into_inner())
        .clone()
        .expect("observation");
    assert_eq!(key, "auth");
    assert_eq!(timeout, HOME_REFRESH_OPERATION_TIMEOUT);
    assert_eq!(
        serde_json::from_slice::<serde_json::Value>(&payload).expect("json"),
        serde_json::json!({
            "type": "refresh",
            "auth_index": "auth-1",
            "access_token_sha256": "abc123"
        })
    );
}

#[test]
fn empty_fingerprint_is_omitted_for_legacy_callers() {
    let request = super::requests::RefreshRequest {
        request_type: "refresh".into(),
        auth_index: "auth-1".into(),
        access_token_sha256: String::new(),
    };
    let value = serde_json::to_value(request).expect("json");
    assert!(value.get("access_token_sha256").is_none());
}

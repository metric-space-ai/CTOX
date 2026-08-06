// ref: internal/home/client_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::certificate::{
    certificate_fingerprint_pem, encode_resp_array, normalize_fingerprint, parse_home_jwt_claims,
};
use super::client::*;
use super::global::HomeRuntime;
use base64::Engine;
use sha2::Digest;
use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Default)]
struct MemoryTransport {
    values: Mutex<HashMap<String, Vec<u8>>>,
    pushes: Mutex<Vec<(String, Vec<u8>, bool)>>,
    request: Mutex<Option<Result<Vec<u8>, TransportFailure>>>,
}
impl HomeTransport for MemoryTransport {
    fn ping(&self) -> Result<(), HomeError> {
        Ok(())
    }
    fn get(&self, key: &str) -> Result<Option<Vec<u8>>, HomeError> {
        Ok(self.values.lock().unwrap().get(key).cloned())
    }
    fn set(&self, key: &str, value: &[u8], options: KvSetOptions) -> Result<bool, HomeError> {
        let mut values = self.values.lock().unwrap();
        if options.nx && values.contains_key(key) {
            return Ok(false);
        }
        if options.xx && !values.contains_key(key) {
            return Ok(false);
        }
        values.insert(key.into(), value.into());
        Ok(true)
    }
    fn compare_and_swap(
        &self,
        key: &str,
        expected: Option<&[u8]>,
        value: &[u8],
        _: Duration,
    ) -> Result<bool, HomeError> {
        let mut values = self.values.lock().unwrap();
        if values.get(key).map(Vec::as_slice) != expected {
            return Ok(false);
        }
        values.insert(key.into(), value.into());
        Ok(true)
    }
    fn delete(&self, keys: &[String]) -> Result<i64, HomeError> {
        let mut values = self.values.lock().unwrap();
        Ok(keys
            .iter()
            .filter(|key| values.remove(*key).is_some())
            .count() as i64)
    }
    fn expire(&self, key: &str, _: Duration) -> Result<bool, HomeError> {
        Ok(self.values.lock().unwrap().contains_key(key))
    }
    fn ttl(&self, key: &str) -> Result<Option<Duration>, HomeError> {
        Ok(self
            .values
            .lock()
            .unwrap()
            .contains_key(key)
            .then_some(Duration::from_secs(1)))
    }
    fn increment(&self, key: &str, delta: i64) -> Result<i64, HomeError> {
        let mut values = self.values.lock().unwrap();
        let next = values
            .get(key)
            .and_then(|v| std::str::from_utf8(v).ok())
            .and_then(|v| v.parse().ok())
            .unwrap_or(0)
            + delta;
        values.insert(key.into(), next.to_string().into_bytes());
        Ok(next)
    }
    fn push(&self, key: &str, payload: &[u8], right: bool) -> Result<(), HomeError> {
        self.pushes
            .lock()
            .unwrap()
            .push((key.into(), payload.into(), right));
        Ok(())
    }
    fn request(&self, _: &str, _: &[u8]) -> Result<Vec<u8>, TransportFailure> {
        self.request
            .lock()
            .unwrap()
            .take()
            .unwrap_or_else(|| Ok(br#"{"ok":true}"#.to_vec()))
    }
    fn request_with_timeout(
        &self,
        key: &str,
        payload: &[u8],
        _: Duration,
    ) -> Result<Vec<u8>, TransportFailure> {
        self.request(key, payload)
    }
}
fn client(transport: Arc<MemoryTransport>) -> Client {
    let facade: Arc<dyn HomeTransport> = transport;
    Client::new(
        HomeConfig {
            enabled: true,
            host: "home".into(),
            port: 6379,
            ..HomeConfig::default()
        },
        facade,
    )
}

#[test]
fn dispatch_request_defaults_count_normalizes_headers_and_carries_policy() {
    let request = new_auth_dispatch_request(
        " model ",
        " session ",
        BTreeMap::from([("Authorization".into(), "secret".into())]),
        0,
        " strict ",
    );
    assert_eq!(request.count, 1);
    assert_eq!(request.model, "model");
    assert_eq!(request.headers.get("authorization").unwrap(), "secret");
    assert_eq!(request.credential_policy, "strict");
}

#[test]
fn kv_set_arguments_validate_conflicts_and_ceil_durations() {
    let args = build_kv_set_args(
        "key",
        b"value",
        KvSetOptions {
            px: Duration::from_micros(1001),
            nx: true,
            ..KvSetOptions::default()
        },
    )
    .unwrap();
    assert!(args.contains(&serde_json::Value::from(2)));
    assert!(build_kv_set_args(
        "key",
        b"v",
        KvSetOptions {
            nx: true,
            xx: true,
            ..KvSetOptions::default()
        }
    )
    .is_err());
}

#[test]
fn kv_conditions_cas_and_dedicated_queue_keys_are_preserved() {
    let transport = Arc::new(MemoryTransport::default());
    let client = client(Arc::clone(&transport));
    assert!(client.kv_set_nx("k", b"v", Duration::ZERO).unwrap());
    assert!(!client.kv_set_nx("k", b"other", Duration::ZERO).unwrap());
    assert!(!client
        .kv_compare_and_swap("k", Some(b"wrong"), b"next", Duration::ZERO)
        .unwrap());
    assert!(client
        .kv_compare_and_swap("k", Some(b"v"), b"next", Duration::ZERO)
        .unwrap());
    client.push_in_flight_snapshot(b"snapshot").unwrap();
    client.push_plugin_status(b"status").unwrap();
    let pushes = transport.pushes.lock().unwrap();
    assert_eq!(pushes[0].0, KEY_IN_FLIGHT_SNAPSHOT);
    assert_eq!(pushes[1].0, KEY_PLUGIN_STATUS);
    assert!(pushes[1].2);
}

#[test]
fn post_send_dispatch_failure_is_ambiguous_and_fences_lifetime() {
    let transport = Arc::new(MemoryTransport::default());
    *transport.request.lock().unwrap() = Some(Err(TransportFailure {
        stage: DispatchFailureStage::AfterSend,
        message: "closed".into(),
    }));
    let client = client(transport);
    let error = client
        .rpop_auth("m", "", BTreeMap::new(), 1, "")
        .unwrap_err();
    assert!(is_ambiguous_dispatch_error(&error));
    assert!(client.ambiguous_dispatch());
    assert_eq!(
        client
            .rpop_auth("m", "", BTreeMap::new(), 1, "")
            .unwrap_err(),
        HomeError::DispatchFenced
    );
    assert!(!client.takeover_eligible());
}

#[test]
fn lifetime_preserves_membership_identity_and_legacy_takeover_suppression() {
    let transport = Arc::new(MemoryTransport::default());
    let client = client(transport);
    client.mark_membership_takeover_eligible();
    let next = client.new_lifetime();
    assert_eq!(
        next.membership_instance_id(),
        client.membership_instance_id()
    );
    assert!(next.takeover_eligible());
    next.enable_legacy_membership();
    assert!(next.legacy_membership());
    assert!(!next.takeover_eligible());
}

#[test]
fn protocol_error_classification_matches_upstream_strings() {
    assert!(is_membership_takeover_unavailable_error(
        "ERR membership_takeover_unavailable"
    ));
    assert!(is_legacy_membership_protocol_error(
        "ERR wrong number of arguments for 'subscribe' command"
    ));
    assert!(is_home_command_unsupported("ERR unknown command CAS"));
}

#[test]
fn certificate_claim_fingerprint_and_resp_protocol_are_stable() {
    let payload = serde_json::json!({"certificate_id":"cert","cluster_id":"cluster","ca_fingerprint":"aa:bb","enrollment_secret":"secret","ip":"127.0.0.1","port":6379,"iat":1});
    let jwt = format!(
        "e30.{}.sig",
        base64::engine::general_purpose::URL_SAFE_NO_PAD
            .encode(serde_json::to_vec(&payload).unwrap())
    );
    let claims = parse_home_jwt_claims(&jwt).unwrap();
    assert_eq!(claims.certificate_id, "cert");
    assert_eq!(normalize_fingerprint(&claims.ca_fingerprint), "aabb");
    let der = b"fake-certificate-der";
    let pem = format!(
        "-----BEGIN CERTIFICATE-----\n{}\n-----END CERTIFICATE-----\n",
        base64::engine::general_purpose::STANDARD.encode(der)
    );
    assert_eq!(
        certificate_fingerprint_pem(pem.as_bytes()).unwrap(),
        format!("{:x}", sha2::Sha256::digest(der))
    );
    assert_eq!(
        encode_resp_array(&["CERTIFICATE".into(), "token".into()]),
        b"*2\r\n$11\r\nCERTIFICATE\r\n$5\r\ntoken\r\n"
    );
}

#[test]
fn injected_runtime_is_instance_isolated_and_clear_if_is_identity_safe() {
    let transport = Arc::new(MemoryTransport::default());
    let first = Arc::new(client(Arc::clone(&transport)));
    let second = Arc::new(client(transport));
    let runtime = HomeRuntime::default();
    runtime.set_current(Arc::clone(&first));
    assert!(!runtime.clear_current_if(&second));
    assert!(Arc::ptr_eq(&runtime.current().unwrap(), &first));
    assert!(runtime.clear_current_if(&first));
    assert!(runtime.current().is_none());
}

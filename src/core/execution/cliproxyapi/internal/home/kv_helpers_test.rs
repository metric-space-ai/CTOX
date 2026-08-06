// ref: internal/home/kv_helpers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::client::*;
use super::global::HomeRuntime;
use super::kv_helpers::*;
use std::sync::Arc;
use std::time::Duration;

struct FailingTransport;
impl HomeTransport for FailingTransport {
    fn ping(&self) -> Result<(), HomeError> {
        Ok(())
    }
    fn get(&self, _: &str) -> Result<Option<Vec<u8>>, HomeError> {
        Err(HomeError::Transport("failed".into()))
    }
    fn set(&self, _: &str, _: &[u8], _: KvSetOptions) -> Result<bool, HomeError> {
        Err(HomeError::Transport("failed".into()))
    }
    fn compare_and_swap(
        &self,
        _: &str,
        _: Option<&[u8]>,
        _: &[u8],
        _: Duration,
    ) -> Result<bool, HomeError> {
        Err(HomeError::Transport("failed".into()))
    }
    fn delete(&self, _: &[String]) -> Result<i64, HomeError> {
        Err(HomeError::Transport("failed".into()))
    }
    fn expire(&self, _: &str, _: Duration) -> Result<bool, HomeError> {
        Err(HomeError::Transport("failed".into()))
    }
    fn ttl(&self, _: &str) -> Result<Option<Duration>, HomeError> {
        Err(HomeError::Transport("failed".into()))
    }
    fn increment(&self, _: &str, _: i64) -> Result<i64, HomeError> {
        Err(HomeError::Transport("failed".into()))
    }
    fn push(&self, _: &str, _: &[u8], _: bool) -> Result<(), HomeError> {
        Err(HomeError::Transport("failed".into()))
    }
    fn request(&self, _: &str, _: &[u8]) -> Result<Vec<u8>, TransportFailure> {
        Err(TransportFailure {
            stage: DispatchFailureStage::BeforeSend,
            message: "failed".into(),
        })
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

#[test]
fn hash_and_log_prefix_do_not_disclose_keys() {
    let hash = hash_key_part("secret-value");
    assert_eq!(hash.len(), 64);
    assert!(!hash.contains("secret"));
    assert_eq!(kv_log_prefix("cpa:test:secret-key"), "cpa:test:*");
    assert_eq!(kv_log_prefix(""), "unknown");
}

#[test]
fn required_helpers_distinguish_non_home_disabled_and_not_ready() {
    let runtime = HomeRuntime::default();
    assert_eq!(
        kv_get_json_required::<serde_json::Value>(&runtime, "key").unwrap(),
        (false, None)
    );
    let transport: Arc<dyn HomeTransport> = Arc::new(FailingTransport);
    let disabled = Arc::new(Client::new(HomeConfig::default(), Arc::clone(&transport)));
    runtime.set_current(disabled);
    assert_eq!(
        current_kv_client(&runtime).unwrap_err(),
        HomeError::Disabled
    );
    let ready = Arc::new(Client::new(
        HomeConfig {
            enabled: true,
            ..HomeConfig::default()
        },
        transport,
    ));
    runtime.set_current(Arc::clone(&ready));
    assert_eq!(
        current_kv_client(&runtime).unwrap_err(),
        HomeError::NotConnected
    );
}

#[test]
fn required_propagates_and_best_effort_swallows_transport_failure() {
    let runtime = HomeRuntime::default();
    let transport: Arc<dyn HomeTransport> = Arc::new(FailingTransport);
    let client = Arc::new(Client::new(
        HomeConfig {
            enabled: true,
            ..HomeConfig::default()
        },
        transport,
    ));
    client.set_heartbeat(true);
    runtime.set_current(client);
    assert!(kv_set_json_required(
        &runtime,
        "cpa:test:key",
        &serde_json::json!({"value":"secret"}),
        Duration::ZERO
    )
    .is_err());
    assert!(!kv_set_json_best_effort(
        &runtime,
        "cpa:test:key",
        &serde_json::json!({"value":"secret"}),
        Duration::ZERO
    ));
}

#[test]
fn ttl_options_match_upstream_ex_semantics() {
    assert_eq!(
        kv_set_options_for_ttl(Duration::ZERO),
        KvSetOptions::default()
    );
    assert_eq!(
        kv_set_options_for_ttl(Duration::from_secs(2)).ex,
        Duration::from_secs(2)
    );
}

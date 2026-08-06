// ref: sdk/access/registry_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex, OnceLock};

use serde_json::json;

use super::*;

struct TestProvider {
    id: String,
}

impl TestProvider {
    fn shared(id: &str) -> SharedProvider {
        Arc::new(Self { id: id.to_owned() })
    }
}

impl Provider for TestProvider {
    fn identifier(&self) -> &str {
        &self.id
    }

    fn authenticate<'a>(&'a self, _request: &'a mut Request) -> AuthenticationFuture<'a> {
        Box::pin(async move {
            AuthenticationOutcome::success(Some(Result {
                provider: self.id.clone(),
                principal: self.id.clone(),
                metadata: None,
            }))
        })
    }
}

fn registry_test_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn remove_test_providers() {
    for key in [
        "access-test-a",
        "access-test-b",
        "access-missing",
        "access-thread-0",
        "access-thread-1",
        "access-thread-2",
        "access-thread-3",
    ] {
        unregister_provider(key);
    }
    clear_exclusive_provider();
}

#[test]
fn registered_providers_returns_only_exclusive_provider() {
    let _guard = registry_test_lock();
    remove_test_providers();
    register_provider("access-test-a", Some(TestProvider::shared("access-test-a")));
    register_provider("access-test-b", Some(TestProvider::shared("access-test-b")));
    set_exclusive_provider("access-test-b");

    let providers = registered_providers().expect("registered snapshot");
    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0].identifier(), "access-test-b");
    remove_test_providers();
}

#[test]
fn registered_providers_restores_all_providers_after_exclusive_cleared() {
    let _guard = registry_test_lock();
    remove_test_providers();
    register_provider("access-test-a", Some(TestProvider::shared("access-test-a")));
    register_provider("access-test-b", Some(TestProvider::shared("access-test-b")));
    set_exclusive_provider("access-test-b");
    clear_exclusive_provider();

    let providers = registered_providers().expect("registered snapshot");
    let ids: Vec<_> = providers
        .iter()
        .map(|provider| provider.identifier())
        .collect();
    assert_eq!(ids, ["access-test-a", "access-test-b"]);
    remove_test_providers();
}

#[test]
fn registered_providers_ignores_stale_exclusive_provider() {
    let _guard = registry_test_lock();
    remove_test_providers();
    register_provider("access-test-a", Some(TestProvider::shared("access-test-a")));
    set_exclusive_provider("access-missing");

    let providers = registered_providers().expect("registered snapshot");
    assert_eq!(providers.len(), 1);
    assert_eq!(providers[0].identifier(), "access-test-a");
    remove_test_providers();
}

#[test]
fn replacement_preserves_order_and_replaces_provider_identity() {
    let _guard = registry_test_lock();
    remove_test_providers();
    let first_a = TestProvider::shared("first-a");
    let replacement_a = TestProvider::shared("replacement-a");
    register_provider(" access-test-a ", Some(Arc::clone(&first_a)));
    register_provider("access-test-b", Some(TestProvider::shared("test-b")));
    register_provider("access-test-a", Some(Arc::clone(&replacement_a)));

    let providers = registered_providers().expect("registered snapshot");
    assert_eq!(providers[0].identifier(), "replacement-a");
    assert_eq!(providers[1].identifier(), "test-b");
    assert!(Arc::ptr_eq(&providers[0], &replacement_a));
    assert!(!Arc::ptr_eq(&providers[0], &first_a));
    remove_test_providers();
}

#[test]
fn unregister_then_reregister_appends_provider_to_order() {
    let _guard = registry_test_lock();
    remove_test_providers();
    register_provider("access-test-a", Some(TestProvider::shared("test-a")));
    register_provider("access-test-b", Some(TestProvider::shared("test-b")));
    unregister_provider(" access-test-a ");
    register_provider("access-test-a", Some(TestProvider::shared("test-a-2")));

    let providers = registered_providers().expect("registered snapshot");
    let ids: Vec<_> = providers
        .iter()
        .map(|provider| provider.identifier())
        .collect();
    assert_eq!(ids, ["test-b", "test-a-2"]);
    remove_test_providers();
}

#[test]
fn blank_type_and_nil_provider_are_ignored() {
    let _guard = registry_test_lock();
    remove_test_providers();
    register_provider("   ", Some(TestProvider::shared("blank")));
    register_provider("access-test-a", None);
    assert!(registered_providers().is_none());
}

#[test]
fn concurrent_registration_keeps_every_provider_and_a_consistent_order() {
    let _guard = registry_test_lock();
    remove_test_providers();
    let threads: Vec<_> = (0..4)
        .map(|index| {
            std::thread::spawn(move || {
                let key = format!("access-thread-{index}");
                register_provider(&key, Some(TestProvider::shared(&key)));
            })
        })
        .collect();
    for thread in threads {
        thread.join().expect("registration thread");
    }

    let providers = registered_providers().expect("registered snapshot");
    let ids: std::collections::BTreeSet<_> = providers
        .iter()
        .map(|provider| provider.identifier())
        .filter(|identifier| identifier.starts_with("access-thread-"))
        .collect();
    assert_eq!(ids.len(), 4);
    assert_eq!(providers.len(), 4);
    remove_test_providers();
}

#[tokio::test]
async fn provider_contract_preserves_nil_result_and_owned_request_body() {
    let provider = TestProvider::shared("access-test-a");
    let mut request = Request {
        method: "POST".to_owned(),
        url: Some("https://example.test/v1".to_owned()),
        headers: Some(Headers::from([(
            "Authorization".to_owned(),
            vec!["Bearer secret".to_owned()],
        )])),
        body: Some(b"payload".to_vec()),
    };
    let outcome = provider.authenticate(&mut request).await;
    assert!(outcome.error.is_none());
    let result = outcome.result.expect("non-nil result");
    assert_eq!(result.provider, "access-test-a");
    assert_eq!(request.body.as_deref(), Some(b"payload".as_slice()));
}

#[test]
fn authentication_outcome_preserves_all_go_nil_combinations() {
    let empty = AuthenticationOutcome::default();
    assert!(empty.result.is_none());
    assert!(empty.error.is_none());

    let both = AuthenticationOutcome {
        result: Some(Result::default()),
        error: Some(new_not_handled_error()),
    };
    assert!(both.result.is_some());
    assert!(both.error.is_some());
}

#[derive(Debug)]
struct Cause;

impl fmt::Display for Cause {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("root cause")
    }
}

impl Error for Cause {}

#[test]
fn auth_error_contract_matches_upstream_fallbacks_and_causes() {
    let missing = new_no_credentials_error();
    assert_eq!(missing.code, AUTH_ERROR_CODE_NO_CREDENTIALS);
    assert_eq!(missing.to_string(), "Missing API key");
    assert_eq!(missing.http_status_code(), HTTP_STATUS_UNAUTHORIZED);
    assert!(is_auth_error_code(
        Some(&missing),
        &AUTH_ERROR_CODE_NO_CREDENTIALS
    ));
    assert!(!is_auth_error_code(None, &AUTH_ERROR_CODE_NO_CREDENTIALS));
    assert_eq!(auth_error_message(None), "");
    assert!(auth_error_cause(None).is_none());
    assert_eq!(
        auth_error_http_status_code(None),
        HTTP_STATUS_INTERNAL_SERVER_ERROR
    );

    let not_handled = new_not_handled_error();
    assert_eq!(
        not_handled.http_status_code(),
        HTTP_STATUS_INTERNAL_SERVER_ERROR
    );

    let internal = new_internal_auth_error("  ", Some(Arc::new(Cause)));
    assert_eq!(internal.code, AUTH_ERROR_CODE_INTERNAL);
    assert_eq!(
        internal.to_string(),
        "Authentication service error: root cause"
    );
    assert!(!format!("{internal:?}").contains("root cause"));
    assert!(internal.source().is_some());
    let zero = AuthError::default();
    assert_eq!(zero.to_string(), "authentication error");
    assert_eq!(zero.http_status_code(), HTTP_STATUS_INTERNAL_SERVER_ERROR);
    assert_eq!(AuthErrorCode::default().as_str(), "");
    assert_eq!(AuthErrorCode::from("future_code").as_str(), "future_code");
}

#[test]
fn access_config_preserves_nil_collections_and_go_omitempty_wire_shape() {
    let nil_config = AccessConfig::default();
    assert!(nil_config.providers.is_none());
    assert_eq!(serde_json::to_value(&nil_config).unwrap(), json!({}));

    let empty_config = AccessConfig {
        providers: Some(Vec::new()),
    };
    assert!(empty_config.providers.is_some());
    assert_eq!(serde_json::to_value(&empty_config).unwrap(), json!({}));

    let decoded: AccessConfig = serde_json::from_value(json!({})).unwrap();
    assert!(decoded.providers.is_none());
}

#[test]
fn access_provider_uses_exact_json_and_yaml_field_names() {
    let provider = AccessProvider {
        name: "custom".to_owned(),
        provider_type: "sdk-type".to_owned(),
        sdk: "example-sdk".to_owned(),
        api_keys: Some(vec!["key".to_owned()]),
        config: Some(std::collections::BTreeMap::from([(
            "enabled".to_owned(),
            json!(true),
        )])),
    };
    let value = serde_json::to_value(&provider).unwrap();
    assert_eq!(value["name"], "custom");
    assert_eq!(value["type"], "sdk-type");
    assert_eq!(value["api-keys"], json!(["key"]));
    assert!(value.get("api_keys").is_none());

    let yaml = serde_yaml::to_string(&provider).unwrap();
    assert!(yaml.contains("api-keys:"));
    let decoded: AccessProvider = serde_yaml::from_str(&yaml).unwrap();
    assert_eq!(decoded, provider);
}

#[test]
fn inline_provider_rejects_empty_keys_and_clones_nonempty_keys() {
    assert!(make_inline_api_key_provider(&[]).is_none());
    let mut keys = vec!["first".to_owned(), "second".to_owned()];
    let provider = make_inline_api_key_provider(&keys).expect("provider");
    keys[0].clear();
    assert_eq!(provider.name, DEFAULT_ACCESS_PROVIDER_NAME);
    assert_eq!(provider.provider_type, ACCESS_PROVIDER_TYPE_CONFIG_API_KEY);
    assert_eq!(
        provider.api_keys.as_deref(),
        Some(["first".to_owned(), "second".to_owned()].as_slice())
    );
    assert!(provider.config.is_none());
}

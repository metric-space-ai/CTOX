// ref: internal/runtime/executor/helps/proxy_helpers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::{
    new_proxy_aware_http_client, ProxyClientFailureSink, ProxyTransportSource, MAX_PROXY_URL_BYTES,
};
use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::proxyutil::{new_direct_transport, HttpTransportRoute, ProxyErrorKind};

#[derive(Default)]
struct Failures(Mutex<Vec<ProxyErrorKind>>);

impl ProxyClientFailureSink for Failures {
    fn on_proxy_build_failure(&self, kind: ProxyErrorKind) {
        self.0.lock().unwrap().push(kind);
    }
}

fn auth(proxy_url: &str) -> Auth {
    let mut auth = Auth::default();
    auth.proxy_url = proxy_url.to_owned();
    auth
}

#[test]
fn direct_auth_bypasses_global_proxy() {
    let plan = new_proxy_aware_http_client(
        Some("http://global-proxy.example.com:8080"),
        Some(&auth("direct")),
        None,
        Duration::ZERO,
        None,
    );
    assert_eq!(plan.source(), ProxyTransportSource::Auth);
    assert_eq!(plan.transport().route(), &HttpTransportRoute::Direct);
    assert!(plan.transport().proxy_url().is_none());
    assert!(!plan.transport().uses_environment_proxy());
    assert_eq!(plan.timeout(), None);
}

#[test]
fn priority_timeout_and_injected_fallback_are_explicit() {
    let plan = new_proxy_aware_http_client(
        Some("http://config.example:8080"),
        Some(&auth(" socks5://auth.example:1080 ")),
        None,
        Duration::from_secs(7),
        None,
    );
    assert_eq!(plan.source(), ProxyTransportSource::Auth);
    assert_eq!(plan.transport().route(), &HttpTransportRoute::Socks5);
    assert_eq!(plan.timeout(), Some(Duration::from_secs(7)));

    let credentialed = new_proxy_aware_http_client(
        None,
        Some(&auth("http://secret-user:secret-pass@proxy.example:8080")),
        None,
        Duration::ZERO,
        None,
    );
    let debug = format!("{credentialed:?}");
    assert!(!debug.contains("secret-user"));
    assert!(!debug.contains("secret-pass"));

    let injected = Arc::new(new_direct_transport());
    let failures = Failures::default();
    let plan = new_proxy_aware_http_client(
        Some("contains-secret@invalid"),
        None,
        Some(Arc::clone(&injected)),
        Duration::ZERO,
        Some(&failures),
    );
    assert_eq!(plan.source(), ProxyTransportSource::Injected);
    assert!(Arc::ptr_eq(plan.transport(), &injected));
    assert_eq!(failures.0.lock().unwrap().len(), 1);
    let debug = format!("{plan:?}");
    assert!(!debug.contains("secret"));
    assert!(!debug.contains("invalid"));
}

#[test]
fn blank_and_oversized_values_never_consult_ambient_proxy_state() {
    let plan =
        new_proxy_aware_http_client(Some("  "), Some(&auth("\t")), None, Duration::ZERO, None);
    assert_eq!(plan.source(), ProxyTransportSource::DirectFallback);
    assert!(!plan.transport().uses_environment_proxy());

    let failures = Failures::default();
    let oversized = format!("http://{}", "x".repeat(MAX_PROXY_URL_BYTES));
    let plan = new_proxy_aware_http_client(
        Some(&oversized),
        None,
        None,
        Duration::ZERO,
        Some(&failures),
    );
    assert_eq!(plan.source(), ProxyTransportSource::DirectFallback);
    assert_eq!(
        failures.0.lock().unwrap().as_slice(),
        &[ProxyErrorKind::InvalidEndpoint]
    );
}

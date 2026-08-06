// ref: sdk/cliproxy/rtprovider_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::proxyutil::{HttpTransportRoute, ProxyErrorKind};

use super::rtprovider::{
    new_default_round_tripper_provider, DefaultRoundTripperProvider, ProxyBuildFailureSink,
};

fn auth(proxy_url: &str) -> Auth {
    let mut auth = Auth::default();
    auth.proxy_url = proxy_url.to_owned();
    auth
}

#[test]
fn round_tripper_for_direct_bypasses_proxy() {
    let provider = new_default_round_tripper_provider();
    let transport = provider
        .round_tripper_for(Some(&auth("direct")))
        .expect("direct transport");

    assert_eq!(transport.route(), &HttpTransportRoute::Direct);
    assert!(transport.proxy_url().is_none());
    assert!(!transport.uses_environment_proxy());
}

#[test]
fn missing_and_blank_proxy_return_no_override() {
    let provider = new_default_round_tripper_provider();
    assert!(provider.round_tripper_for(None).is_none());
    assert!(provider.round_tripper_for(Some(&auth("  "))).is_none());
    assert_eq!(provider.cached_transport_count(), 0);
}

#[test]
fn trimmed_proxy_value_is_cached_by_identity() {
    let provider = new_default_round_tripper_provider();
    let first = provider
        .round_tripper_for(Some(&auth(" http://proxy.example:8080 ")))
        .expect("proxy transport");
    let second = provider
        .round_tripper_for(Some(&auth("http://proxy.example:8080")))
        .expect("cached proxy transport");

    assert!(Arc::ptr_eq(&first, &second));
    assert_eq!(first.route(), &HttpTransportRoute::HttpProxy);
    assert_eq!(provider.cached_transport_count(), 1);
}

#[derive(Default)]
struct FailureSink(Mutex<Vec<ProxyErrorKind>>);

impl ProxyBuildFailureSink for FailureSink {
    fn on_proxy_build_failure(&self, kind: ProxyErrorKind) {
        self.0.lock().unwrap().push(kind);
    }
}

#[test]
fn invalid_proxy_reports_only_typed_failure_and_is_not_cached() {
    let sink = Arc::new(FailureSink::default());
    let provider = DefaultRoundTripperProvider::with_failure_sink(sink.clone());

    assert!(provider
        .round_tripper_for(Some(&auth("secret-user:secret-pass@invalid")))
        .is_none());
    assert_eq!(provider.cached_transport_count(), 0);
    assert_eq!(
        sink.0.lock().unwrap().as_slice(),
        &[ProxyErrorKind::MissingSchemeOrHost]
    );
}

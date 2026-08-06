// ref: internal/util/proxy.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::proxyutil::{HttpTransport, HttpTransportRoute, ProxyErrorKind};

use super::{set_proxy, ProxyTransportTarget};

#[derive(Default)]
struct Client {
    transport: Option<HttpTransport>,
    installs: usize,
}

impl ProxyTransportTarget for Client {
    fn install_proxy_transport(&mut self, transport: HttpTransport) {
        self.transport = Some(transport);
        self.installs += 1;
    }
}

#[test]
fn nil_guards_leave_client_untouched() {
    let mut client = Client::default();
    set_proxy(None, Some(&mut client)).unwrap();
    set_proxy::<Client>(Some("http://proxy.example:8080"), None).unwrap();
    assert_eq!(client.installs, 0);
}

#[test]
fn delegates_all_modes_to_canonical_proxyutil() {
    for (raw, route) in [
        ("", HttpTransportRoute::Direct),
        ("direct", HttpTransportRoute::Direct),
        ("http://proxy.example:8080", HttpTransportRoute::HttpProxy),
        ("socks5://proxy.example:1080", HttpTransportRoute::Socks5),
    ] {
        let mut client = Client::default();
        set_proxy(Some(raw), Some(&mut client)).unwrap();
        let transport = client.transport.unwrap();
        assert_eq!(transport.route(), &route, "raw={raw:?}");
        assert!(!transport.uses_environment_proxy());
    }
}

#[test]
fn invalid_proxy_is_typed_and_does_not_replace_existing_transport() {
    let mut client = Client::default();
    set_proxy(Some("direct"), Some(&mut client)).unwrap();
    let error = set_proxy(Some("http://user:secret%@proxy.example"), Some(&mut client))
        .expect_err("invalid URL should fail");
    assert_eq!(error.kind(), ProxyErrorKind::ParseUrl);
    assert_eq!(client.installs, 1);
    assert_eq!(
        client.transport.unwrap().route(),
        &HttpTransportRoute::Direct
    );
    assert!(!error.to_string().contains("secret"));
}

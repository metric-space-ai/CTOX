// ref: internal/util/proxy.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::proxyutil::{build_http_transport, HttpTransport, ProxyError};

/// Rust HTTP clients implement this narrow seam to accept the canonical proxy
/// transport. This avoids creating a second proxy parser or dialer authority in
/// `internal::util`.
pub trait ProxyTransportTarget {
    fn install_proxy_transport(&mut self, transport: HttpTransport);
}

/// Applies the configured proxy to an HTTP-client adapter.
///
/// `None` for either argument mirrors Go's nil guards and leaves the target
/// untouched. `Some("")` is distinct: the canonical CTOX proxy utility
/// installs an explicit direct transport so ambient environment proxies cannot
/// leak into runtime behavior. Unlike Go's log-and-continue helper, Rust
/// returns a typed error and leaves the target unchanged on invalid input.
pub fn set_proxy<T: ProxyTransportTarget>(
    proxy_url: Option<&str>,
    http_client: Option<&mut T>,
) -> Result<(), ProxyError> {
    let (Some(proxy_url), Some(http_client)) = (proxy_url, http_client) else {
        return Ok(());
    };
    let (transport, _) = build_http_transport(proxy_url)?;
    http_client.install_proxy_transport(transport);
    Ok(())
}

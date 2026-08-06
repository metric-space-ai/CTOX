// ref: internal/runtime/executor/helps/proxy_helpers.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::proxyutil::{
    build_http_transport, new_direct_transport, HttpTransport, ProxyErrorKind,
};

pub const MAX_PROXY_URL_BYTES: usize = 8 * 1024;

/// Credential-safe proxy build diagnostics. Raw URLs are deliberately absent.
pub trait ProxyClientFailureSink: Send + Sync {
    fn on_proxy_build_failure(&self, kind: ProxyErrorKind);
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProxyTransportSource {
    Auth,
    Config,
    Injected,
    DirectFallback,
}

/// Transport and timeout selected for a caller-owned HTTP client. CTOX keeps
/// client construction in the host adapter, while this helper owns upstream's
/// priority/fallback semantics.
#[derive(Clone)]
pub struct ProxyAwareHttpClientPlan {
    transport: Arc<HttpTransport>,
    timeout: Option<Duration>,
    source: ProxyTransportSource,
}

impl ProxyAwareHttpClientPlan {
    #[must_use]
    pub fn transport(&self) -> &Arc<HttpTransport> {
        &self.transport
    }

    #[must_use]
    pub const fn timeout(&self) -> Option<Duration> {
        self.timeout
    }

    #[must_use]
    pub const fn source(&self) -> ProxyTransportSource {
        self.source
    }
}

impl fmt::Debug for ProxyAwareHttpClientPlan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProxyAwareHttpClientPlan")
            .field("transport", &self.transport)
            .field("timeout", &self.timeout)
            .field("source", &self.source)
            .finish()
    }
}

/// Selects a proxy-aware transport with upstream priority:
/// auth override, typed runtime config, injected request transport, then an
/// explicit direct fallback. Invalid configured proxies fall through without
/// exposing their URL or credentials.
#[must_use]
pub fn new_proxy_aware_http_client(
    config_proxy_url: Option<&str>,
    auth: Option<&Auth>,
    injected_transport: Option<Arc<HttpTransport>>,
    timeout: Duration,
    failure_sink: Option<&dyn ProxyClientFailureSink>,
) -> ProxyAwareHttpClientPlan {
    let auth_proxy = auth
        .map(|auth| auth.proxy_url.trim())
        .filter(|proxy| !proxy.is_empty());
    let config_proxy = config_proxy_url
        .map(str::trim)
        .filter(|proxy| !proxy.is_empty());
    let configured = auth_proxy
        .map(|proxy| (proxy, ProxyTransportSource::Auth))
        .or_else(|| config_proxy.map(|proxy| (proxy, ProxyTransportSource::Config)));

    if let Some((proxy, source)) = configured {
        let built = if proxy.len() > MAX_PROXY_URL_BYTES {
            Err(ProxyErrorKind::InvalidEndpoint)
        } else {
            build_http_transport(proxy)
                .map(|(transport, _)| transport)
                .map_err(|error| error.kind())
        };
        match built {
            Ok(transport) => {
                return ProxyAwareHttpClientPlan {
                    transport: Arc::new(transport),
                    timeout: (!timeout.is_zero()).then_some(timeout),
                    source,
                };
            }
            Err(kind) => {
                if let Some(sink) = failure_sink {
                    sink.on_proxy_build_failure(kind);
                }
            }
        }
    }

    let (transport, source) = injected_transport.map_or_else(
        || {
            (
                Arc::new(new_direct_transport()),
                ProxyTransportSource::DirectFallback,
            )
        },
        |transport| (transport, ProxyTransportSource::Injected),
    );
    ProxyAwareHttpClientPlan {
        transport,
        timeout: (!timeout.is_zero()).then_some(timeout),
        source,
    }
}

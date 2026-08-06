// ref: internal/runtime/executor/helps/utls_client.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: the host transport factory owns the real TLS handshake and session cache.
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::sync::Arc;
use std::time::Duration;

use crate::sdk::pluginapi::{
    HostHttpClient, HttpRequest, HttpResponse, HttpStreamResponse, PluginExecutionError,
    PluginFuture,
};

use super::claude_upstream::is_anthropic_upstream_url;

pub const CLAUDE_CODE_SESSION_CACHE_CAPACITY: usize = 32;
pub const CLAUDE_CODE_ROUND_TRIPPER_CACHE_CAPACITY: usize = 64;
pub const UTLS_PROTECTED_HOSTS: [&str; 2] = ["api.anthropic.com", "chatgpt.com"];

pub const CLAUDE_CODE_CIPHER_SUITES: [u16; 17] = [
    0x1301, 0x1302, 0x1303, 0xC02B, 0xC02F, 0xC02C, 0xC030, 0xCCA9, 0xCCA8, 0xC009, 0xC013, 0xC00A,
    0xC014, 0x009C, 0x009D, 0x002F, 0x0035,
];

pub const CLAUDE_CODE_TLS_EXTENSIONS: [&str; 15] = [
    "server_name",
    "extended_master_secret",
    "renegotiation_info",
    "supported_groups:x25519,p256,p384",
    "ec_point_formats:uncompressed",
    "session_ticket",
    "alpn:http/1.1",
    "status_request",
    "signature_algorithms:ecdsa_p256_sha256,pss_sha256,pkcs1_sha256,ecdsa_p384_sha384,pss_sha384,pkcs1_sha384,pss_sha512,pkcs1_sha512,pkcs1_sha1",
    "signed_certificate_timestamp",
    "key_share:x25519",
    "psk_key_exchange_modes:dhe",
    "supported_versions:tls13,tls12",
    "boring_padding",
    "pre_shared_key",
];
pub const CLAUDE_CODE_OMIT_EMPTY_PSK: bool = true;
pub const CLAUDE_CODE_SKIP_RESUMPTION_WITHOUT_PSK_EXTENSION: bool = true;

pub const CLAUDE_CODE_MESSAGES_HEADER_ORDER: [&str; 22] = [
    "Accept",
    "Authorization",
    "Content-Type",
    "User-Agent",
    "X-Claude-Code-Session-Id",
    "X-Stainless-Arch",
    "X-Stainless-Lang",
    "X-Stainless-OS",
    "X-Stainless-Package-Version",
    "X-Stainless-Retry-Count",
    "X-Stainless-Runtime",
    "X-Stainless-Runtime-Version",
    "X-Stainless-Timeout",
    "anthropic-beta",
    "anthropic-dangerous-direct-browser-access",
    "anthropic-version",
    "x-app",
    "x-client-request-id",
    "Connection",
    "Host",
    "Accept-Encoding",
    "Content-Length",
];

pub const CLAUDE_CODE_COUNT_TOKENS_HEADER_ORDER: [&str; 21] = [
    "Accept",
    "Authorization",
    "Content-Type",
    "User-Agent",
    "X-Claude-Code-Session-Id",
    "X-Stainless-Arch",
    "X-Stainless-Lang",
    "X-Stainless-OS",
    "X-Stainless-Package-Version",
    "X-Stainless-Retry-Count",
    "X-Stainless-Runtime",
    "X-Stainless-Runtime-Version",
    "anthropic-beta",
    "anthropic-dangerous-direct-browser-access",
    "anthropic-version",
    "x-app",
    "x-client-request-id",
    "Connection",
    "Host",
    "Accept-Encoding",
    "Content-Length",
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtlsTransportProfile {
    ClaudeCodeNodeOpenSsl,
    Chrome,
    Standard,
}

#[derive(Clone)]
pub struct UtlsTransports {
    pub anthropic: Arc<dyn HostHttpClient>,
    pub chrome: Arc<dyn HostHttpClient>,
    pub standard: Arc<dyn HostHttpClient>,
}

pub trait UtlsTransportFactory: Send + Sync {
    fn build(
        &self,
        proxy_url: Option<&str>,
        profile: UtlsTransportProfile,
    ) -> Result<Arc<dyn HostHttpClient>, UtlsClientError>;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum UtlsClientError {
    Build,
    Timeout,
}

impl fmt::Display for UtlsClientError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Build => "fingerprinted HTTP client construction failed",
            Self::Timeout => "fingerprinted HTTP request timed out",
        })
    }
}

impl std::error::Error for UtlsClientError {}

#[derive(Clone)]
pub struct UtlsHttpClient {
    anthropic: Arc<dyn HostHttpClient>,
    chrome: Arc<dyn HostHttpClient>,
    standard: Arc<dyn HostHttpClient>,
    timeout: Option<Duration>,
}

pub fn claude_code_request_header_order(request_target: &str) -> &'static [&'static str] {
    if request_target.starts_with("/v1/messages/count_tokens") {
        &CLAUDE_CODE_COUNT_TOKENS_HEADER_ORDER
    } else {
        &CLAUDE_CODE_MESSAGES_HEADER_ORDER
    }
}

/// Proxy-scoped session identity. Host transports use this as their bounded
/// resumption/cache key so tickets can never cross proxy boundaries.
pub fn claude_code_transport_scope_key(proxy_url: Option<&str>) -> String {
    let proxy = proxy_url.map(str::trim).unwrap_or_default();
    format!("claude-code-node-openssl\0{proxy}")
}

pub fn new_utls_http_client(
    context_client: Option<Arc<dyn HostHttpClient>>,
    factory: &dyn UtlsTransportFactory,
    proxy_url: Option<&str>,
    timeout: Duration,
) -> Result<UtlsHttpClient, UtlsClientError> {
    let proxy = proxy_url.map(str::trim).filter(|value| !value.is_empty());
    let transports = if proxy.is_none() {
        context_client.map_or_else(
            || build_transports(factory, None),
            |client| {
                Ok(UtlsTransports {
                    anthropic: client.clone(),
                    chrome: client.clone(),
                    standard: client,
                })
            },
        )?
    } else {
        build_transports(factory, proxy)?
    };
    Ok(UtlsHttpClient {
        anthropic: transports.anthropic,
        chrome: transports.chrome,
        standard: transports.standard,
        timeout: (!timeout.is_zero()).then_some(timeout),
    })
}

fn build_transports(
    factory: &dyn UtlsTransportFactory,
    proxy: Option<&str>,
) -> Result<UtlsTransports, UtlsClientError> {
    Ok(UtlsTransports {
        anthropic: factory.build(proxy, UtlsTransportProfile::ClaudeCodeNodeOpenSsl)?,
        chrome: factory.build(proxy, UtlsTransportProfile::Chrome)?,
        standard: factory.build(proxy, UtlsTransportProfile::Standard)?,
    })
}

impl UtlsHttpClient {
    fn transport_for(&self, url: &str) -> Arc<dyn HostHttpClient> {
        if url::Url::parse(url)
            .ok()
            .as_ref()
            .is_some_and(is_anthropic_upstream_url)
        {
            return self.anthropic.clone();
        }
        let chatgpt = url::Url::parse(url).ok().is_some_and(|url| {
            url.scheme().eq_ignore_ascii_case("https")
                && url
                    .host_str()
                    .is_some_and(|host| host.eq_ignore_ascii_case("chatgpt.com"))
        });
        if chatgpt {
            self.chrome.clone()
        } else {
            self.standard.clone()
        }
    }
}

impl HostHttpClient for UtlsHttpClient {
    fn execute<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpResponse> {
        let client = self.transport_for(&request.url);
        let timeout = self.timeout;
        Box::pin(async move {
            match timeout {
                Some(timeout) => tokio::time::timeout(timeout, client.execute(request))
                    .await
                    .map_err(|_| Arc::new(UtlsClientError::Timeout) as PluginExecutionError)?,
                None => client.execute(request).await,
            }
        })
    }

    fn execute_stream<'a>(&'a self, request: HttpRequest) -> PluginFuture<'a, HttpStreamResponse> {
        let client = self.transport_for(&request.url);
        let timeout = self.timeout;
        Box::pin(async move {
            match timeout {
                Some(timeout) => tokio::time::timeout(timeout, client.execute_stream(request))
                    .await
                    .map_err(|_| Arc::new(UtlsClientError::Timeout) as PluginExecutionError)?,
                None => client.execute_stream(request).await,
            }
        })
    }
}

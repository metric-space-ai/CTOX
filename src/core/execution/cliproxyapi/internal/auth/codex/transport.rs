// ref: internal/auth/codex/openai_auth.go:194-252 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use wreq::header::{ACCEPT, CONTENT_TYPE};
use wreq::{Client, Proxy};

use super::openai_auth::{
    CodexCodeExchangeTransport, CodexExchangeHttpResponse, CodexExchangeRequest,
    CodexRefreshHttpResponse, CodexRefreshRequest, CodexRefreshTransport,
    CodexRefreshTransportFailure, TOKEN_URL,
};

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone)]
pub struct CodexHttpTransport {
    client: Client,
    endpoint: String,
    proxy_mode: CodexProxyMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexProxyMode {
    Direct,
    Proxy,
}

#[derive(Clone, Copy)]
pub enum CodexProxyOverride<'a> {
    Inherit,
    Direct,
    Proxy(&'a super::token::SecretString),
}

pub fn new_codex_transport_with_proxy<'a>(
    configured_proxy: Option<&'a super::token::SecretString>,
    proxy_override: CodexProxyOverride<'a>,
) -> Result<CodexHttpTransport, CodexTransportBuildError> {
    let proxy_url = match proxy_override {
        CodexProxyOverride::Inherit => configured_proxy.map(|proxy| proxy.expose_secret()),
        CodexProxyOverride::Direct => Some("direct"),
        CodexProxyOverride::Proxy(proxy) => Some(proxy.expose_secret()),
    };
    CodexHttpTransport::new(proxy_url)
}

impl CodexHttpTransport {
    /// Environment proxy discovery is disabled. A proxy must arrive through
    /// typed host configuration.
    pub fn new(proxy_url: Option<&str>) -> Result<Self, CodexTransportBuildError> {
        Self::with_endpoint(TOKEN_URL, proxy_url)
    }

    pub(crate) fn with_endpoint(
        endpoint: &str,
        proxy_url: Option<&str>,
    ) -> Result<Self, CodexTransportBuildError> {
        if endpoint.trim().is_empty() {
            return Err(CodexTransportBuildError::InvalidEndpoint);
        }
        let proxy_mode = classify_proxy_mode(proxy_url)?;
        let mut builder = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .retry(wreq::retry::Policy::never())
            .redirect(wreq::redirect::Policy::none());
        match proxy_mode {
            CodexProxyMode::Proxy => {
                let proxy_url = proxy_url
                    .map(str::trim)
                    .filter(|value| !value.is_empty())
                    .ok_or(CodexTransportBuildError::InvalidProxy)?;
                let proxy =
                    Proxy::all(proxy_url).map_err(|_| CodexTransportBuildError::InvalidProxy)?;
                builder = builder.proxy(proxy);
            }
            CodexProxyMode::Direct => builder = builder.no_proxy(),
        }
        let client = builder
            .build()
            .map_err(|_| CodexTransportBuildError::Client)?;
        Ok(Self {
            client,
            endpoint: endpoint.to_owned(),
            proxy_mode,
        })
    }

    pub fn proxy_mode(&self) -> CodexProxyMode {
        self.proxy_mode
    }
}

fn classify_proxy_mode(
    proxy_url: Option<&str>,
) -> Result<CodexProxyMode, CodexTransportBuildError> {
    let Some(proxy_url) = proxy_url.map(str::trim).filter(|value| !value.is_empty()) else {
        return Ok(CodexProxyMode::Direct);
    };
    if proxy_url.eq_ignore_ascii_case("direct") || proxy_url.eq_ignore_ascii_case("none") {
        return Ok(CodexProxyMode::Direct);
    }
    let parsed = url::Url::parse(proxy_url).map_err(|_| CodexTransportBuildError::InvalidProxy)?;
    if !matches!(parsed.scheme(), "socks5" | "socks5h" | "http" | "https") {
        return Err(CodexTransportBuildError::InvalidProxy);
    }
    Proxy::all(proxy_url).map_err(|_| CodexTransportBuildError::InvalidProxy)?;
    Ok(CodexProxyMode::Proxy)
}

impl fmt::Debug for CodexHttpTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CodexHttpTransport")
            .field("endpoint", &self.endpoint)
            .field("proxy_mode", &self.proxy_mode)
            .field("credentials", &"[REDACTED]")
            .finish()
    }
}

impl CodexRefreshTransport for CodexHttpTransport {
    fn execute<'a>(
        &'a self,
        request: &'a CodexRefreshRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<CodexRefreshHttpResponse, CodexRefreshTransportFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let body = request.form_body();
            let response = self
                .client
                .post(&self.endpoint)
                .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
                .header(ACCEPT, "application/json")
                .timeout(timeout)
                .body(body.as_slice().to_vec())
                .send()
                .await
                .map_err(classify_transport_error)?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .await
                .map_err(classify_transport_error)?
                .to_vec();
            Ok(CodexRefreshHttpResponse::new(status, body))
        })
    }
}

impl CodexCodeExchangeTransport for CodexHttpTransport {
    fn exchange<'a>(
        &'a self,
        request: &'a CodexExchangeRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<CodexExchangeHttpResponse, CodexRefreshTransportFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let body = request.form_body();
            let response = self
                .client
                .post(&self.endpoint)
                .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
                .header(ACCEPT, "application/json")
                .timeout(timeout)
                .body(body.as_slice().to_vec())
                .send()
                .await
                .map_err(classify_transport_error)?;
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .await
                .map_err(classify_transport_error)?
                .to_vec();
            Ok(CodexExchangeHttpResponse::new(status, body))
        })
    }
}

fn classify_transport_error(error: wreq::Error) -> CodexRefreshTransportFailure {
    if error.is_timeout() {
        CodexRefreshTransportFailure::Timeout
    } else if error.is_connect() {
        CodexRefreshTransportFailure::Connect
    } else {
        CodexRefreshTransportFailure::Protocol
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CodexTransportBuildError {
    InvalidEndpoint,
    InvalidProxy,
    Client,
}

impl fmt::Display for CodexTransportBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidEndpoint => "Codex token endpoint is invalid",
            Self::InvalidProxy => "Codex proxy configuration is invalid",
            Self::Client => "Codex HTTP transport could not be built",
        })
    }
}

impl std::error::Error for CodexTransportBuildError {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::internal::auth::codex::{
        CodexRefreshCoordinator, CodexStoredCredentials, SecretString, SystemRefreshClock,
    };

    #[test]
    fn invalid_proxy_never_echoes_embedded_credentials() {
        let raw = "not a proxy://operator:do-not-leak";
        let error = CodexHttpTransport::new(Some(raw)).unwrap_err();
        let rendered = format!("{error:?} {error}");
        assert!(!rendered.contains("operator"));
        assert!(!rendered.contains("do-not-leak"));
    }

    #[tokio::test]
    async fn loopback_refresh_uses_form_contract_and_no_ambient_proxy() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/oauth/token", listener.local_addr().unwrap());
        let captured = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let server_capture = Arc::clone(&captured);
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 2048];
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                if read == 0 {
                    break;
                }
                request.extend_from_slice(&buffer[..read]);
                if let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") {
                    let end = end + 4;
                    let headers = String::from_utf8_lossy(&request[..end]);
                    let length = headers
                        .lines()
                        .find_map(|line| {
                            line.to_ascii_lowercase()
                                .strip_prefix("content-length: ")
                                .and_then(|value| value.parse::<usize>().ok())
                        })
                        .unwrap_or(0);
                    if request.len() >= end + length {
                        break;
                    }
                }
            }
            *server_capture.lock().await = request;
            let body = br#"{"access_token":"new-access","expires_in":3600}"#;
            let headers = format!(
                "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            );
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(body).await.unwrap();
        });

        let current = CodexStoredCredentials::new(
            SecretString::new("old-id").unwrap(),
            SecretString::new("old-access").unwrap(),
            SecretString::new("old-refresh").unwrap(),
        );
        let token = CodexRefreshCoordinator::default()
            .refresh(
                &CodexHttpTransport::with_endpoint(&endpoint, None).unwrap(),
                &SystemRefreshClock,
                current,
                1,
            )
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(token.access_token().expose_secret(), "new-access");

        let request = String::from_utf8(captured.lock().await.clone()).unwrap();
        let lower = request.to_ascii_lowercase();
        assert!(request.starts_with("POST /oauth/token HTTP/1.1\r\n"));
        assert!(lower.contains("content-type: application/x-www-form-urlencoded"));
        assert!(lower.contains("accept: application/json"));
        assert!(request.contains("refresh_token=old-refresh"));
        assert!(!lower.contains("sec-fetch"));
    }
}

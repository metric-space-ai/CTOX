// ref: internal/runtime/executor/antigravity_executor_auth.go:148-193 @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::future::Future;
use std::pin::Pin;
use std::time::Duration;

use wreq::header::{CONTENT_TYPE, HOST, USER_AGENT};
use wreq::{Client, Proxy};

use super::{
    AntigravityFlowTransport, AntigravityHttpFuture, AntigravityHttpMethod, AntigravityHttpRequest,
    AntigravityHttpResponse, AntigravityHttpTransportFailure, AntigravityRefreshHttpResponse,
    AntigravityRefreshRequest, AntigravityRefreshTransport, AntigravityRefreshTransportFailure,
    TOKEN_ENDPOINT,
};
use crate::sdk::auth::LoginCancellation;

const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Clone)]
pub struct AntigravityHttpTransport {
    client: Client,
    endpoint: String,
}

impl AntigravityHttpTransport {
    pub fn new(proxy_url: Option<&str>) -> Result<Self, AntigravityTransportBuildError> {
        Self::with_endpoint(TOKEN_ENDPOINT, proxy_url)
    }

    fn with_endpoint(
        endpoint: &str,
        proxy_url: Option<&str>,
    ) -> Result<Self, AntigravityTransportBuildError> {
        if endpoint.trim().is_empty() {
            return Err(AntigravityTransportBuildError::InvalidEndpoint);
        }
        let mut builder = Client::builder()
            .connect_timeout(CONNECT_TIMEOUT)
            .retry(wreq::retry::Policy::never())
            .redirect(wreq::redirect::Policy::none());
        match proxy_url.map(str::trim).filter(|value| !value.is_empty()) {
            Some(proxy_url) => {
                builder = builder.proxy(
                    Proxy::all(proxy_url)
                        .map_err(|_| AntigravityTransportBuildError::InvalidProxy)?,
                );
            }
            None => builder = builder.no_proxy(),
        }
        Ok(Self {
            client: builder
                .build()
                .map_err(|_| AntigravityTransportBuildError::Client)?,
            endpoint: endpoint.to_owned(),
        })
    }
}

impl fmt::Debug for AntigravityHttpTransport {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AntigravityHttpTransport")
            .field("endpoint", &self.endpoint)
            .field("credentials", &"[REDACTED]")
            .finish()
    }
}

impl AntigravityRefreshTransport for AntigravityHttpTransport {
    fn execute<'a>(
        &'a self,
        request: &'a AntigravityRefreshRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<
                    Output = Result<
                        AntigravityRefreshHttpResponse,
                        AntigravityRefreshTransportFailure,
                    >,
                > + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            let body = request.form_body();
            let response = self
                .client
                .post(&self.endpoint)
                .header(HOST, "oauth2.googleapis.com")
                .header(CONTENT_TYPE, "application/x-www-form-urlencoded")
                .header(USER_AGENT, "Go-http-client/2.0")
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
            Ok(AntigravityRefreshHttpResponse::new(status, body))
        })
    }
}

impl AntigravityFlowTransport for AntigravityHttpTransport {
    fn execute<'a>(
        &'a self,
        request: &'a AntigravityHttpRequest,
        timeout: Duration,
        cancellation: &'a LoginCancellation,
    ) -> AntigravityHttpFuture<'a> {
        Box::pin(async move {
            let mut builder = match request.method {
                AntigravityHttpMethod::Get => self.client.get(&request.url),
                AntigravityHttpMethod::Post => self.client.post(&request.url),
            };
            for (name, value) in &request.headers {
                builder = builder.header(name, value);
            }
            builder = builder.timeout(timeout);
            if !request.body.is_empty() {
                builder = builder.body(request.body.as_slice().to_vec());
            }
            let response = tokio::select! {
                response = builder.send() => response.map_err(classify_flow_transport_error)?,
                () = cancellation.cancelled() => {
                    return Err(AntigravityHttpTransportFailure::Cancelled);
                }
            };
            let status = response.status().as_u16();
            let body = response
                .bytes()
                .await
                .map_err(classify_flow_transport_error)?
                .to_vec();
            Ok(AntigravityHttpResponse::new(status, body))
        })
    }
}

fn classify_transport_error(error: wreq::Error) -> AntigravityRefreshTransportFailure {
    if error.is_timeout() {
        AntigravityRefreshTransportFailure::Timeout
    } else if error.is_connect() {
        AntigravityRefreshTransportFailure::Connect
    } else {
        AntigravityRefreshTransportFailure::Protocol
    }
}

fn classify_flow_transport_error(error: wreq::Error) -> AntigravityHttpTransportFailure {
    if error.is_timeout() {
        AntigravityHttpTransportFailure::Timeout
    } else if error.is_connect() {
        AntigravityHttpTransportFailure::Connect
    } else {
        AntigravityHttpTransportFailure::Protocol
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AntigravityTransportBuildError {
    InvalidEndpoint,
    InvalidProxy,
    Client,
}

impl fmt::Display for AntigravityTransportBuildError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::InvalidEndpoint => "Antigravity token endpoint is invalid",
            Self::InvalidProxy => "Antigravity proxy configuration is invalid",
            Self::Client => "Antigravity HTTP transport could not be built",
        })
    }
}

impl std::error::Error for AntigravityTransportBuildError {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::time::SystemTime;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::*;
    use crate::internal::auth::antigravity::{
        AntigravityRefreshCoordinator, AntigravityStoredCredentials, SecretString,
    };

    #[test]
    fn invalid_proxy_never_echoes_credentials() {
        let raw = "not a proxy://operator:do-not-leak";
        let error = AntigravityHttpTransport::new(Some(raw)).unwrap_err();
        let rendered = format!("{error:?} {error}");
        assert!(!rendered.contains("operator"));
        assert!(!rendered.contains("do-not-leak"));
    }

    #[tokio::test]
    async fn loopback_refresh_matches_google_form_and_user_agent() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}/token", listener.local_addr().unwrap());
        let captured = Arc::new(tokio::sync::Mutex::new(Vec::new()));
        let server_capture = Arc::clone(&captured);
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await.unwrap();
            let mut request = Vec::new();
            let mut buffer = [0_u8; 2048];
            loop {
                let read = socket.read(&mut buffer).await.unwrap();
                request.extend_from_slice(&buffer[..read]);
                let Some(end) = request.windows(4).position(|part| part == b"\r\n\r\n") else {
                    continue;
                };
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
            *server_capture.lock().await = request;
            let body =
                br#"{"access_token":"access-new","refresh_token":"refresh-new","expires_in":3600}"#;
            let headers = format!("HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            socket.write_all(headers.as_bytes()).await.unwrap();
            socket.write_all(body).await.unwrap();
        });
        let now = SystemTime::UNIX_EPOCH + Duration::from_secs(10_000);
        let current = AntigravityStoredCredentials::new(
            SecretString::new("access-old").unwrap(),
            SecretString::new("refresh-old").unwrap(),
            now,
            "project-1",
        )
        .unwrap();
        let rotated = AntigravityRefreshCoordinator::default()
            .refresh(
                &AntigravityHttpTransport::with_endpoint(&endpoint, None).unwrap(),
                current,
                now,
            )
            .await
            .unwrap();
        server.await.unwrap();
        assert_eq!(rotated.access_token().expose_secret(), "access-new");
        assert_eq!(rotated.project_id(), "project-1");
        let request = String::from_utf8(captured.lock().await.clone()).unwrap();
        let lower = request.to_ascii_lowercase();
        assert!(request.starts_with("POST /token HTTP/1.1\r\n"));
        assert!(lower.contains("host: oauth2.googleapis.com"));
        assert!(lower.contains("user-agent: go-http-client/2.0"));
        assert!(lower.contains("content-type: application/x-www-form-urlencoded"));
        assert!(request.contains("refresh_token=refresh-old"));
        assert!(request.contains("grant_type=refresh_token"));
    }
}

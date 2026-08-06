// ref: internal/auth/claude/oauth_server.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::fmt;
use std::io;
use std::net::{Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

use crate::internal::auth::loopback_http::{
    read_request_head, response, serve_bounded_loopback, write_response, HttpResponse, RequestHead,
    IO_TIMEOUT,
};

use super::errors::{
    new_authentication_error, AuthenticationError, ERR_CALLBACK_TIMEOUT, ERR_INVALID_STATE,
    ERR_PORT_IN_USE, ERR_SERVER_START_FAILED,
};
use super::html_templates::{render_login_success_html, HtmlTemplateError};
use super::token::SecretString;

const DEFAULT_PLATFORM_URL: &str = "https://console.anthropic.com/";
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Clone, PartialEq, Eq)]
pub struct OAuthResult {
    code: Option<SecretString>,
    state: Option<SecretString>,
    error: Option<String>,
}

impl OAuthResult {
    fn success(code: SecretString, state: SecretString) -> Self {
        Self {
            code: Some(code),
            state: Some(state),
            error: None,
        }
    }

    fn failure(error: impl Into<String>) -> Self {
        Self {
            code: None,
            state: None,
            error: Some(error.into()),
        }
    }

    pub fn code(&self) -> Option<&SecretString> {
        self.code.as_ref()
    }

    pub fn state(&self) -> Option<&SecretString> {
        self.state.as_ref()
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn is_success(&self) -> bool {
        self.error.is_none() && self.code.is_some() && self.state.is_some()
    }
}

impl fmt::Debug for OAuthResult {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("OAuthResult")
            .field("code", &self.code.as_ref().map(|_| "[REDACTED]"))
            .field("state", &self.state.as_ref().map(|_| "[REDACTED]"))
            .field("error", &self.error.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

#[derive(Debug)]
pub enum OAuthServerError {
    Authentication(AuthenticationError),
    AlreadyRunning,
    NotRunning,
    ResultChannelClosed,
    Shutdown,
}

impl OAuthServerError {
    pub fn authentication(&self) -> Option<&AuthenticationError> {
        match self {
            Self::Authentication(error) => Some(error),
            _ => None,
        }
    }

    fn authentication_error(base: &AuthenticationError, cause: ServerCause) -> Self {
        Self::Authentication(new_authentication_error(base, cause))
    }
}

impl fmt::Display for OAuthServerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Authentication(error) => error.fmt(formatter),
            Self::AlreadyRunning => formatter.write_str("OAuth callback server is already running"),
            Self::NotRunning => formatter.write_str("OAuth callback server is not running"),
            Self::ResultChannelClosed => {
                formatter.write_str("OAuth callback result channel closed")
            }
            Self::Shutdown => formatter.write_str("OAuth callback server shutdown failed"),
        }
    }
}

impl std::error::Error for OAuthServerError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Authentication(error) => Some(error),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum ServerCause {
    PortInUse,
    Bind,
    Accept,
    StateMismatch,
    CallbackTimeout,
}

impl fmt::Display for ServerCause {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::PortInUse => "loopback port is already in use",
            Self::Bind => "failed to bind loopback listener",
            Self::Accept => "loopback listener failed",
            Self::StateMismatch => "OAuth state mismatch",
            Self::CallbackTimeout => "timeout waiting for OAuth callback",
        })
    }
}

impl std::error::Error for ServerCause {}

pub struct OAuthServer {
    port: u16,
    expected_state: SecretString,
    local_addr: Option<SocketAddr>,
    result_rx: Option<mpsc::Receiver<Result<OAuthResult, OAuthServerError>>>,
    fatal_rx: Option<mpsc::Receiver<OAuthServerError>>,
    shutdown_tx: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
    running: Arc<AtomicBool>,
}

impl OAuthServer {
    pub fn new(port: u16, expected_state: SecretString) -> Self {
        Self {
            port,
            expected_state,
            local_addr: None,
            result_rx: None,
            fatal_rx: None,
            shutdown_tx: None,
            task: None,
            running: Arc::new(AtomicBool::new(false)),
        }
    }

    pub async fn start(&mut self) -> Result<(), OAuthServerError> {
        if self.is_running() {
            return Err(OAuthServerError::AlreadyRunning);
        }

        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, self.port))
            .await
            .map_err(|error| {
                if error.kind() == io::ErrorKind::AddrInUse {
                    OAuthServerError::authentication_error(&ERR_PORT_IN_USE, ServerCause::PortInUse)
                } else {
                    OAuthServerError::authentication_error(
                        &ERR_SERVER_START_FAILED,
                        ServerCause::Bind,
                    )
                }
            })?;
        let local_addr = listener.local_addr().map_err(|_| {
            OAuthServerError::authentication_error(&ERR_SERVER_START_FAILED, ServerCause::Bind)
        })?;

        let (result_tx, result_rx) = mpsc::channel(1);
        let (fatal_tx, fatal_rx) = mpsc::channel(1);
        let (shutdown_tx, shutdown_rx) = oneshot::channel();
        let running = Arc::clone(&self.running);
        running.store(true, Ordering::Release);
        let expected_state = self.expected_state.clone();
        let task = tokio::spawn(async move {
            let _running_guard = RunningGuard(running);
            serve(listener, expected_state, result_tx, fatal_tx, shutdown_rx).await;
        });

        self.local_addr = Some(local_addr);
        self.result_rx = Some(result_rx);
        self.fatal_rx = Some(fatal_rx);
        self.shutdown_tx = Some(shutdown_tx);
        self.task = Some(task);
        Ok(())
    }

    pub fn local_addr(&self) -> Option<SocketAddr> {
        self.local_addr
    }

    pub fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }

    pub async fn wait_for_callback(
        &mut self,
        timeout: Duration,
    ) -> Result<OAuthResult, OAuthServerError> {
        if !self.is_running() {
            return Err(OAuthServerError::NotRunning);
        }
        let result_rx = self
            .result_rx
            .as_mut()
            .ok_or(OAuthServerError::NotRunning)?;
        let fatal_rx = self.fatal_rx.as_mut().ok_or(OAuthServerError::NotRunning)?;

        tokio::select! {
            result = result_rx.recv() => result.ok_or(OAuthServerError::ResultChannelClosed)?,
            fatal = fatal_rx.recv() => Err(fatal.unwrap_or(OAuthServerError::ResultChannelClosed)),
            _ = tokio::time::sleep(timeout) => Err(OAuthServerError::authentication_error(
                &ERR_CALLBACK_TIMEOUT,
                ServerCause::CallbackTimeout,
            )),
        }
    }

    pub async fn stop(&mut self) -> Result<(), OAuthServerError> {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(mut task) = self.task.take() {
            if tokio::time::timeout(SHUTDOWN_TIMEOUT, &mut task)
                .await
                .is_err()
            {
                task.abort();
                self.running.store(false, Ordering::Release);
                return Err(OAuthServerError::Shutdown);
            }
        }
        self.running.store(false, Ordering::Release);
        self.local_addr = None;
        self.result_rx = None;
        self.fatal_rx = None;
        Ok(())
    }
}

impl Drop for OAuthServer {
    fn drop(&mut self) {
        if let Some(shutdown_tx) = self.shutdown_tx.take() {
            let _ = shutdown_tx.send(());
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
        self.running.store(false, Ordering::Release);
    }
}

struct RunningGuard(Arc<AtomicBool>);

impl Drop for RunningGuard {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

async fn serve(
    listener: TcpListener,
    expected_state: SecretString,
    result_tx: mpsc::Sender<Result<OAuthResult, OAuthServerError>>,
    fatal_tx: mpsc::Sender<OAuthServerError>,
    shutdown_rx: oneshot::Receiver<()>,
) {
    let expected_state = Arc::new(expected_state);
    let handler = {
        let expected_state = Arc::clone(&expected_state);
        move |stream| {
            let expected_state = Arc::clone(&expected_state);
            let result_tx = result_tx.clone();
            async move {
                let _ = handle_connection(stream, &expected_state, &result_tx).await;
            }
        }
    };
    serve_bounded_loopback(
        listener,
        fatal_tx,
        shutdown_rx,
        || OAuthServerError::authentication_error(&ERR_SERVER_START_FAILED, ServerCause::Accept),
        handler,
    )
    .await;
}

async fn handle_connection(
    mut stream: TcpStream,
    expected_state: &SecretString,
    result_tx: &mpsc::Sender<Result<OAuthResult, OAuthServerError>>,
) -> io::Result<()> {
    let request = match tokio::time::timeout(IO_TIMEOUT, read_request_head(&mut stream)).await {
        Ok(Ok(request)) => request,
        Ok(Err(_)) => {
            return write_response(
                &mut stream,
                response(
                    400,
                    "Bad Request",
                    "text/plain; charset=utf-8",
                    b"Bad request".to_vec(),
                ),
            )
            .await;
        }
        Err(_) => {
            return write_response(
                &mut stream,
                response(
                    408,
                    "Request Timeout",
                    "text/plain; charset=utf-8",
                    b"Request timeout".to_vec(),
                ),
            )
            .await;
        }
    };

    let response = route_request(&request, expected_state, result_tx);
    write_response(&mut stream, response).await
}

fn route_request(
    request: &RequestHead,
    expected_state: &SecretString,
    result_tx: &mpsc::Sender<Result<OAuthResult, OAuthServerError>>,
) -> HttpResponse {
    if request.method != "GET" {
        return response(
            405,
            "Method Not Allowed",
            "text/plain; charset=utf-8",
            b"Method not allowed".to_vec(),
        )
        .with_header("Allow", "GET");
    }

    let url = match url::Url::parse(&format!("http://localhost{}", request.target)) {
        Ok(url) if url.host_str() == Some("localhost") => url,
        _ => {
            return response(
                400,
                "Bad Request",
                "text/plain; charset=utf-8",
                b"Bad request".to_vec(),
            );
        }
    };

    match url.path() {
        "/callback" => callback_response(&url, expected_state, result_tx),
        "/success" => success_response(&url),
        _ => response(
            404,
            "Not Found",
            "text/plain; charset=utf-8",
            b"Not found".to_vec(),
        ),
    }
}

fn callback_response(
    url: &url::Url,
    expected_state: &SecretString,
    result_tx: &mpsc::Sender<Result<OAuthResult, OAuthServerError>>,
) -> HttpResponse {
    let query = |name: &str| {
        url.query_pairs()
            .find_map(|(key, value)| (key == name).then(|| value.into_owned()))
            .unwrap_or_default()
    };
    let provider_error = query("error");
    if !provider_error.is_empty() {
        let _ = result_tx.try_send(Ok(OAuthResult::failure(provider_error)));
        return response(
            400,
            "Bad Request",
            "text/plain; charset=utf-8",
            b"OAuth error received".to_vec(),
        );
    }

    let code = query("code");
    if code.is_empty() {
        let _ = result_tx.try_send(Ok(OAuthResult::failure("no_code")));
        return response(
            400,
            "Bad Request",
            "text/plain; charset=utf-8",
            b"No authorization code received".to_vec(),
        );
    }
    let state = query("state");
    if state.is_empty() {
        let _ = result_tx.try_send(Ok(OAuthResult::failure("no_state")));
        return response(
            400,
            "Bad Request",
            "text/plain; charset=utf-8",
            b"No state parameter received".to_vec(),
        );
    }
    if !state_matches(expected_state.expose_secret(), &state) {
        let _ = result_tx.try_send(Err(OAuthServerError::authentication_error(
            &ERR_INVALID_STATE,
            ServerCause::StateMismatch,
        )));
        return response(
            400,
            "Bad Request",
            "text/plain; charset=utf-8",
            b"Invalid OAuth state".to_vec(),
        );
    }

    let code = match SecretString::new(code) {
        Ok(code) => code,
        Err(_) => unreachable!("non-empty callback code was already checked"),
    };
    let state = match SecretString::new(state) {
        Ok(state) => state,
        Err(_) => unreachable!("non-empty callback state was already checked"),
    };
    let _ = result_tx.try_send(Ok(OAuthResult::success(code, state)));
    response(302, "Found", "text/plain; charset=utf-8", Vec::new())
        .with_header("Location", "/success")
}

fn state_matches(expected: &str, received: &str) -> bool {
    let expected_digest = Sha256::digest(expected.as_bytes());
    let received_digest = Sha256::digest(received.as_bytes());
    bool::from(expected_digest.ct_eq(&received_digest))
}

fn success_response(url: &url::Url) -> HttpResponse {
    let setup_required = url
        .query_pairs()
        .any(|(key, value)| key == "setup_required" && value == "true");
    let platform_url = url
        .query_pairs()
        .find_map(|(key, value)| (key == "platform_url").then(|| value.into_owned()))
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| DEFAULT_PLATFORM_URL.to_owned());

    match render_login_success_html(setup_required, &platform_url) {
        Ok(html) => response(
            200,
            "OK",
            "text/html; charset=utf-8",
            html.into_bytes(),
        )
        .with_header(
            "Content-Security-Policy",
            "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; base-uri 'none'; frame-ancestors 'none'",
        ),
        Err(HtmlTemplateError::InvalidPlatformUrl
        | HtmlTemplateError::UnsupportedPlatformUrlScheme
        | HtmlTemplateError::PlatformUrlContainsControlCharacter) => response(
            400,
            "Bad Request",
            "text/plain; charset=utf-8",
            b"Invalid platform URL".to_vec(),
        ),
    }
}

#[cfg(test)]
mod tests {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    use super::*;

    async fn start(expected_state: &str) -> OAuthServer {
        let mut server = OAuthServer::new(0, SecretString::new(expected_state).unwrap());
        server.start().await.unwrap();
        assert!(server.is_running());
        assert!(server.local_addr().unwrap().ip().is_loopback());
        server
    }

    async fn request(addr: SocketAddr, request: &[u8]) -> String {
        let mut stream = TcpStream::connect(addr).await.unwrap();
        stream.write_all(request).await.unwrap();
        stream.shutdown().await.unwrap();
        let mut response = Vec::new();
        stream.read_to_end(&mut response).await.unwrap();
        String::from_utf8(response).unwrap()
    }

    #[tokio::test]
    async fn valid_callback_redirects_and_delivers_redacted_secrets() {
        let mut server = start("expected-state-do-not-log").await;
        let addr = server.local_addr().unwrap();
        let response = request(
            addr,
            b"GET /callback?code=authorization-do-not-log&state=expected-state-do-not-log HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 302 Found\r\n"));
        assert!(response.contains("Location: /success\r\n"));

        let result = server
            .wait_for_callback(Duration::from_secs(1))
            .await
            .unwrap();
        assert!(result.is_success());
        assert_eq!(
            result.code().unwrap().expose_secret(),
            "authorization-do-not-log"
        );
        assert_eq!(
            result.state().unwrap().expose_secret(),
            "expected-state-do-not-log"
        );
        let debug = format!("{result:?}");
        assert!(!debug.contains("authorization-do-not-log"));
        assert!(!debug.contains("expected-state-do-not-log"));
        server.stop().await.unwrap();
        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn mismatched_state_is_rejected_at_callback_boundary() {
        let mut server = start("expected-state").await;
        let response = request(
            server.local_addr().unwrap(),
            b"GET /callback?code=secret-code&state=attacker-state HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 400 Bad Request\r\n"));
        assert!(!response.contains("attacker-state"));
        assert!(!response.contains("secret-code"));

        let error = server
            .wait_for_callback(Duration::from_secs(1))
            .await
            .unwrap_err();
        assert_eq!(error.authentication().unwrap().error_type, "invalid_state");
        assert!(!format!("{error:?}").contains("attacker-state"));
        server.stop().await.unwrap();
    }

    #[tokio::test]
    async fn provider_error_and_missing_parameters_match_upstream_results_without_reflection() {
        for (target, expected_error, expected_body) in [
            (
                "/callback?error=%3Cscript%3Ealert(1)%3C%2Fscript%3E",
                "<script>alert(1)</script>",
                "OAuth error received",
            ),
            (
                "/callback?state=expected-state",
                "no_code",
                "No authorization code received",
            ),
            (
                "/callback?code=secret-code",
                "no_state",
                "No state parameter received",
            ),
        ] {
            let mut server = start("expected-state").await;
            let wire = format!("GET {target} HTTP/1.1\r\nHost: localhost\r\n\r\n");
            let response = request(server.local_addr().unwrap(), wire.as_bytes()).await;
            assert!(response.starts_with("HTTP/1.1 400 Bad Request\r\n"));
            assert!(response.contains(expected_body));
            assert!(!response.contains("<script>"));
            assert!(!response.contains("secret-code"));
            let result = server
                .wait_for_callback(Duration::from_secs(1))
                .await
                .unwrap();
            assert_eq!(result.error(), Some(expected_error));
            assert!(!format!("{result:?}").contains(expected_error));
            server.stop().await.unwrap();
        }
    }

    #[tokio::test]
    async fn method_rejection_does_not_complete_callback_and_wait_times_out() {
        let mut server = start("expected-state").await;
        let response = request(
            server.local_addr().unwrap(),
            b"POST /callback?code=secret-code&state=expected-state HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n",
        )
        .await;
        assert!(response.starts_with("HTTP/1.1 405 Method Not Allowed\r\n"));
        assert!(response.contains("Allow: GET\r\n"));
        let error = server
            .wait_for_callback(Duration::from_millis(10))
            .await
            .unwrap_err();
        assert_eq!(
            error.authentication().unwrap().error_type,
            "callback_timeout"
        );
        server.stop().await.unwrap();
    }

    #[tokio::test]
    async fn occupied_port_maps_to_the_upstream_port_in_use_error() {
        let listener = TcpListener::bind((Ipv4Addr::LOCALHOST, 0)).await.unwrap();
        let port = listener.local_addr().unwrap().port();
        let mut server = OAuthServer::new(port, SecretString::new("expected-state").unwrap());
        let error = server.start().await.unwrap_err();
        assert_eq!(error.authentication().unwrap().error_type, "port_in_use");
        assert_eq!(error.authentication().unwrap().code, 13);
        assert!(!server.is_running());
    }

    #[tokio::test]
    async fn success_page_uses_hardened_template_renderer() {
        let mut server = start("expected-state").await;
        let addr = server.local_addr().unwrap();
        let default_page = request(addr, b"GET /success HTTP/1.1\r\nHost: localhost\r\n\r\n").await;
        assert!(default_page.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(default_page.contains("https://console.anthropic.com/"));
        assert!(default_page.contains("Content-Security-Policy:"));

        let setup_page = request(
            addr,
            b"GET /success?setup_required=true&platform_url=https%3A%2F%2Fclaude.ai%2Fsettings%3Fa%3D1%26b%3D2 HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        assert!(setup_page.contains("Additional Setup Required"));
        assert_eq!(
            setup_page
                .matches("https://claude.ai/settings?a=1&amp;b=2")
                .count(),
            2
        );

        let rejected = request(
            addr,
            b"GET /success?platform_url=javascript%3Aalert%281%29 HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        assert!(rejected.starts_with("HTTP/1.1 400 Bad Request\r\n"));
        assert!(!rejected.contains("javascript:"));
        server.stop().await.unwrap();
    }

    #[tokio::test]
    async fn duplicate_callbacks_keep_the_first_result() {
        let mut server = start("expected-state").await;
        let addr = server.local_addr().unwrap();
        let first = request(
            addr,
            b"GET /callback?code=first-code&state=expected-state HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        let second = request(
            addr,
            b"GET /callback?code=second-code&state=expected-state HTTP/1.1\r\nHost: localhost\r\n\r\n",
        )
        .await;
        assert!(first.starts_with("HTTP/1.1 302 Found\r\n"));
        assert!(second.starts_with("HTTP/1.1 302 Found\r\n"));
        let result = server
            .wait_for_callback(Duration::from_secs(1))
            .await
            .unwrap();
        assert_eq!(result.code().unwrap().expose_secret(), "first-code");
        server.stop().await.unwrap();
    }

    #[tokio::test]
    async fn slow_partial_request_does_not_delay_valid_callback() {
        let mut server = start("expected-state").await;
        let addr = server.local_addr().unwrap();

        let mut slow = TcpStream::connect(addr).await.unwrap();
        slow.write_all(
            b"GET /callback?code=slow-code&state=expected-state HTTP/1.1\r\nHost: localhost",
        )
        .await
        .unwrap();

        let valid = tokio::time::timeout(
            Duration::from_millis(250),
            request(
                addr,
                b"GET /callback?code=valid-code&state=expected-state HTTP/1.1\r\nHost: localhost\r\n\r\n",
            ),
        )
        .await
        .expect("one partial loopback request must not serialize the accept loop");
        assert!(valid.starts_with("HTTP/1.1 302 Found\r\n"));

        let result = server
            .wait_for_callback(Duration::from_millis(250))
            .await
            .unwrap();
        assert_eq!(result.code().unwrap().expose_secret(), "valid-code");

        drop(slow);
        server.stop().await.unwrap();
    }
}

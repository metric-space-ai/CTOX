// ref: internal/auth/codex/openai_auth_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use tokio::sync::Semaphore;

use super::openai::PkceCodes;
use super::openai_auth::{
    generate_auth_url, CodexRefreshCoordinator, CodexRefreshError, CodexRefreshHttpResponse,
    CodexRefreshRequest, CodexRefreshTransport, CodexRefreshTransportFailure, RefreshClock,
    REFRESH_TIMEOUT,
};
use super::token::{CodexStoredCredentials, SecretString};

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn credentials(refresh_token: &str) -> CodexStoredCredentials {
    CodexStoredCredentials::new(
        SecretString::new("current-id").unwrap(),
        SecretString::new("current-access").unwrap(),
        SecretString::new(refresh_token).unwrap(),
    )
}

#[derive(Clone, Copy)]
struct FixedClock(SystemTime);

impl RefreshClock for FixedClock {
    fn now(&self) -> SystemTime {
        self.0
    }

    fn sleep(
        &self,
        _duration: Duration,
    ) -> Pin<Box<dyn Future<Output = Result<(), CodexRefreshTransportFailure>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }
}

struct CapturingTransport {
    response: Mutex<Option<Result<CodexRefreshHttpResponse, CodexRefreshTransportFailure>>>,
    calls: AtomicUsize,
    timeouts: Mutex<Vec<Duration>>,
}

impl CapturingTransport {
    fn responding(response: CodexRefreshHttpResponse) -> Self {
        Self {
            response: Mutex::new(Some(Ok(response))),
            calls: AtomicUsize::new(0),
            timeouts: Mutex::new(Vec::new()),
        }
    }
}

impl CodexRefreshTransport for CapturingTransport {
    fn execute<'a>(
        &'a self,
        _request: &'a CodexRefreshRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<CodexRefreshHttpResponse, CodexRefreshTransportFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            lock_recover(&self.timeouts).push(timeout);
            lock_recover(&self.response)
                .take()
                .expect("one captured response")
        })
    }
}

// Extension disposition: the upstream production method is covered here
// alongside its refresh tests so all authorization URL parameters stay pinned.
#[test]
fn auth_url_contains_the_complete_upstream_parameter_set() {
    let pkce = PkceCodes::new(
        SecretString::new("verifier-do-not-log").unwrap(),
        "fixed-challenge",
    )
    .unwrap();
    let url = url::Url::parse(&generate_auth_url("state with space", &pkce)).unwrap();
    let params: HashMap<_, _> = url.query_pairs().into_owned().collect();

    assert_eq!(url.scheme(), "https");
    assert_eq!(url.host_str(), Some("auth.openai.com"));
    assert_eq!(
        params.get("client_id").unwrap(),
        "app_EMoamEEZ73f0CkXaXp7hrann"
    );
    assert_eq!(params.get("response_type").unwrap(), "code");
    assert_eq!(
        params.get("redirect_uri").unwrap(),
        "http://localhost:1455/auth/callback"
    );
    assert_eq!(
        params.get("scope").unwrap(),
        "openid email profile offline_access"
    );
    assert_eq!(params.get("state").unwrap(), "state with space");
    assert_eq!(params.get("code_challenge").unwrap(), "fixed-challenge");
    assert_eq!(params.get("code_challenge_method").unwrap(), "S256");
    assert_eq!(params.get("prompt").unwrap(), "login");
    assert_eq!(params.get("id_token_add_organizations").unwrap(), "true");
    assert_eq!(params.get("codex_cli_simplified_flow").unwrap(), "true");
    assert!(!url.as_str().contains("verifier-do-not-log"));
}

// Disposition: adapted. wreq has no blanket `http.Client.Timeout`; each Codex
// operation supplies an explicit bound. A delayed loopback response proves no
// shorter hidden client timeout interrupts the operation.
#[cfg(feature = "codex-http-transport")]
#[tokio::test]
async fn new_codex_transport_does_not_set_a_hidden_request_timeout() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::transport::CodexHttpTransport;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/oauth/token", listener.local_addr().unwrap());
    let server = tokio::spawn(async move {
        let (mut socket, _) = listener.accept().await.unwrap();
        let mut request = [0_u8; 4096];
        let _ = socket.read(&mut request).await.unwrap();
        tokio::time::sleep(Duration::from_millis(75)).await;
        let body = br#"{"probe":"ok"}"#;
        let headers = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            body.len()
        );
        socket.write_all(headers.as_bytes()).await.unwrap();
        socket.write_all(body).await.unwrap();
    });

    let transport = CodexHttpTransport::with_endpoint(&endpoint, None).unwrap();
    let request = CodexRefreshRequest::new(SecretString::new("timeout-probe").unwrap());
    let started = tokio::time::Instant::now();
    let response = transport
        .execute(&request, Duration::from_millis(500))
        .await
        .unwrap();
    server.await.unwrap();
    assert!(started.elapsed() >= Duration::from_millis(50));
    assert!(format!("{response:?}").contains("status: 200"));
}

// Disposition: ported with CTOX's cancellation adaptation. The durable host
// owns cancellation; the provider refresh receives its independent bound.
#[tokio::test]
async fn refresh_tokens_uses_independent_timeout() {
    let transport = CapturingTransport::responding(CodexRefreshHttpResponse::new(
        400,
        br#"{"error":"probe"}"#.to_vec(),
    ));
    let coordinator = CodexRefreshCoordinator::default();
    let error = coordinator
        .refresh(
            &transport,
            &FixedClock(SystemTime::UNIX_EPOCH),
            credentials("independent-timeout-token"),
            1,
        )
        .await
        .unwrap_err();
    assert_eq!(
        error,
        CodexRefreshError::Http {
            status: 400,
            retryable: true
        }
    );
    assert_eq!(*lock_recover(&transport.timeouts), vec![REFRESH_TIMEOUT]);
}

// Disposition: adapted for the secret boundary. Go includes the provider body
// in its error; Rust classifies `refresh_token_reused` without retaining or
// rendering that credential-bearing response.
#[tokio::test]
async fn non_retryable_refresh_token_reuse_attempts_only_once_and_is_redacted() {
    let transport = CapturingTransport::responding(CodexRefreshHttpResponse::new(
        400,
        br#"{"error":"invalid_grant","code":"refresh_token_reused","detail":"do-not-leak"}"#
            .to_vec(),
    ));
    let error = CodexRefreshCoordinator::default()
        .refresh(
            &transport,
            &FixedClock(SystemTime::UNIX_EPOCH),
            credentials("dummy_refresh_token"),
            3,
        )
        .await
        .unwrap_err();
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        error,
        CodexRefreshError::Http {
            status: 400,
            retryable: false
        }
    );
    let rendered = format!("{error:?} {error}");
    assert!(!rendered.contains("refresh_token_reused"));
    assert!(!rendered.contains("dummy_refresh_token"));
    assert!(!rendered.contains("do-not-leak"));
}

struct BlockingTransport {
    calls: AtomicUsize,
    started: Semaphore,
    released: Semaphore,
}

impl CodexRefreshTransport for BlockingTransport {
    fn execute<'a>(
        &'a self,
        _request: &'a CodexRefreshRequest,
        _timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<CodexRefreshHttpResponse, CodexRefreshTransportFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.add_permits(1);
            self.released.acquire().await.unwrap().forget();
            Ok(CodexRefreshHttpResponse::new(
                200,
                br#"{"access_token":"new-access","refresh_token":"new-refresh","expires_in":3600}"#
                    .to_vec(),
            ))
        })
    }
}

// Disposition: adapted to CTOX ownership. Upstream uses package-global
// singleflight across CodexAuth instances; CTOX injects one host-owned
// coordinator into all logical instances, preserving the same deduplication
// without process-global mutable authority.
#[tokio::test]
async fn refresh_deduplicates_across_callers_sharing_the_host_coordinator() {
    let transport = Arc::new(BlockingTransport {
        calls: AtomicUsize::new(0),
        started: Semaphore::new(0),
        released: Semaphore::new(0),
    });
    let coordinator = Arc::new(CodexRefreshCoordinator::default());
    let run = |coordinator: Arc<CodexRefreshCoordinator>, transport: Arc<BlockingTransport>| {
        tokio::spawn(async move {
            coordinator
                .refresh(
                    transport.as_ref(),
                    &FixedClock(SystemTime::UNIX_EPOCH),
                    credentials("shared-refresh-token"),
                    1,
                )
                .await
        })
    };

    let first = run(Arc::clone(&coordinator), Arc::clone(&transport));
    transport.started.acquire().await.unwrap().forget();
    let second = run(Arc::clone(&coordinator), Arc::clone(&transport));
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    transport.released.add_permits(1);

    let first = first.await.unwrap().unwrap();
    let second = second.await.unwrap().unwrap();
    assert_eq!(first, second);
    assert_eq!(first.access_token().expose_secret(), "new-access");
    assert_eq!(first.refresh_token().expose_secret(), "new-refresh");
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
}

#[cfg(feature = "codex-http-transport")]
mod proxy_tests {
    use super::*;
    use crate::internal::auth::codex::{
        new_codex_transport_with_proxy, CodexProxyMode, CodexProxyOverride,
    };

    // Disposition: ported through the typed proxy boundary.
    #[test]
    fn direct_override_disables_configured_proxy() {
        let configured = SecretString::new("http://proxy.example.com:8080").unwrap();
        let transport =
            new_codex_transport_with_proxy(Some(&configured), CodexProxyOverride::Direct).unwrap();
        assert_eq!(transport.proxy_mode(), CodexProxyMode::Direct);
    }

    // Disposition: ported. An invalid configured URL deliberately proves that
    // the valid override is selected first without exposing either URL.
    #[test]
    fn proxy_override_takes_precedence_over_configured_proxy() {
        let configured = SecretString::new("ftp://configured.invalid").unwrap();
        let override_proxy = SecretString::new("http://override.example.com:8081").unwrap();
        let transport = new_codex_transport_with_proxy(
            Some(&configured),
            CodexProxyOverride::Proxy(&override_proxy),
        )
        .unwrap();
        assert_eq!(transport.proxy_mode(), CodexProxyMode::Proxy);
    }
}

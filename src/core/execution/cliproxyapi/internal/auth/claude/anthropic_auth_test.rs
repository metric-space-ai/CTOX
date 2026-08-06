// ref: internal/auth/claude/anthropic_auth_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime};

use tokio::sync::Semaphore;

use super::anthropic_auth::{
    ClaudeAuth, ClaudeCodeExchangeTransport, ClaudeRefreshTransport, ExchangeHttpResponse,
    ExchangeRequest, OAuthInspectHttpResponse, OAuthInspectKind, OAuthInspectRequest, RefreshClock,
    RefreshError, RefreshHttpResponse, RefreshRequest, RefreshTransportFailure, AUTH_SCOPE,
    EXCHANGE_TIMEOUT, REFRESH_TIMEOUT,
};
use super::identity::{valid_device_id, CLAUDE_DEVICE_POOL_SIZE};
use super::pkce::PkceCodes;
use super::token::{ClaudeTokenData, SecretString};

fn lock_recover<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct CandidateTransport {
    exchange_body: Vec<u8>,
    refresh_body: Vec<u8>,
    profile_body: Vec<u8>,
    inspect_failure: bool,
    exchange_requests: Mutex<Vec<Vec<u8>>>,
    refresh_requests: Mutex<Vec<Vec<u8>>>,
    inspect_requests: Mutex<Vec<OAuthInspectKind>>,
}

impl CandidateTransport {
    fn successful() -> Self {
        Self {
            exchange_body: br#"{
                "access_token":"candidate-access",
                "refresh_token":"candidate-refresh",
                "token_type":"Bearer",
                "expires_in":3600,
                "account":{"uuid":"token-account","email_address":"token@example.com"},
                "organization":{"uuid":"token-org","name":"Token Org"}
            }"#
            .to_vec(),
            refresh_body: br#"{
                "access_token":"refreshed-access",
                "expires_in":3600
            }"#
            .to_vec(),
            profile_body: br#"{
                "account":{"uuid":"profile-account","email":"profile@example.com"},
                "organization":{"uuid":"profile-org","name":"Profile Org"}
            }"#
            .to_vec(),
            inspect_failure: false,
            exchange_requests: Mutex::new(Vec::new()),
            refresh_requests: Mutex::new(Vec::new()),
            inspect_requests: Mutex::new(Vec::new()),
        }
    }

    fn inspect_response(
        &self,
        request: &OAuthInspectRequest,
    ) -> Result<OAuthInspectHttpResponse, RefreshTransportFailure> {
        lock_recover(&self.inspect_requests).push(request.kind());
        if self.inspect_failure {
            return Err(RefreshTransportFailure::Connect);
        }
        let body = match request.kind() {
            OAuthInspectKind::Profile => self.profile_body.clone(),
            OAuthInspectKind::Roles => br#"{"roles":["claude_cli"]}"#.to_vec(),
        };
        Ok(OAuthInspectHttpResponse::new(200, body))
    }
}

impl ClaudeCodeExchangeTransport for CandidateTransport {
    fn exchange<'a>(
        &'a self,
        request: &'a ExchangeRequest,
        timeout: Duration,
    ) -> Pin<
        Box<dyn Future<Output = Result<ExchangeHttpResponse, RefreshTransportFailure>> + Send + 'a>,
    > {
        Box::pin(async move {
            assert_eq!(timeout, EXCHANGE_TIMEOUT);
            lock_recover(&self.exchange_requests).push(request.json_body().unwrap().to_vec());
            Ok(ExchangeHttpResponse::new(200, self.exchange_body.clone()))
        })
    }

    fn inspect<'a>(
        &'a self,
        request: &'a OAuthInspectRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<OAuthInspectHttpResponse, RefreshTransportFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            assert_eq!(timeout, EXCHANGE_TIMEOUT);
            self.inspect_response(request)
        })
    }
}

impl ClaudeRefreshTransport for CandidateTransport {
    fn execute<'a>(
        &'a self,
        request: &'a RefreshRequest,
        timeout: Duration,
    ) -> Pin<
        Box<dyn Future<Output = Result<RefreshHttpResponse, RefreshTransportFailure>> + Send + 'a>,
    > {
        Box::pin(async move {
            assert_eq!(timeout, REFRESH_TIMEOUT);
            lock_recover(&self.refresh_requests).push(request.json_body().unwrap().to_vec());
            Ok(RefreshHttpResponse::new(
                200,
                None,
                None,
                self.refresh_body.clone(),
            ))
        })
    }

    fn inspect<'a>(
        &'a self,
        request: &'a OAuthInspectRequest,
        timeout: Duration,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<OAuthInspectHttpResponse, RefreshTransportFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            assert_eq!(timeout, REFRESH_TIMEOUT);
            self.inspect_response(request)
        })
    }
}

fn candidate_pkce() -> PkceCodes {
    PkceCodes {
        code_verifier: "candidate-verifier".to_owned(),
        code_challenge: "candidate-challenge".to_owned(),
    }
}

#[tokio::test]
async fn candidate_exchange_persists_profile_identity_and_device_pool() {
    let auth = ClaudeAuth::with_clock(
        CandidateTransport::successful(),
        FixedClock(SystemTime::UNIX_EPOCH),
    );
    let bundle = auth
        .exchange_code_for_tokens(
            &SecretString::new("candidate-code#fragment-state").unwrap(),
            &SecretString::new("original-state").unwrap(),
            &candidate_pkce(),
        )
        .await
        .unwrap();

    assert_eq!(bundle.token_data().email(), "profile@example.com");
    assert_eq!(bundle.token_data().account_uuid(), "profile-account");
    assert_eq!(bundle.token_data().organization_uuid(), "profile-org");
    assert_eq!(bundle.token_data().organization_name(), "Profile Org");
    assert_eq!(bundle.device_ids().len(), CLAUDE_DEVICE_POOL_SIZE);
    assert!(valid_device_id(&bundle.device_ids()[0]));
    assert_eq!(
        *lock_recover(&auth.transport().inspect_requests),
        vec![OAuthInspectKind::Profile, OAuthInspectKind::Roles]
    );

    let request = lock_recover(&auth.transport().exchange_requests)[0].clone();
    assert_eq!(
        String::from_utf8(request).unwrap(),
        r#"{"grant_type":"authorization_code","code":"candidate-code","redirect_uri":"http://localhost:54545/callback","client_id":"9d1c250a-e61b-44d9-88ed-5944d1962f5e","code_verifier":"candidate-verifier","state":"fragment-state"}"#
    );

    let storage = auth.create_token_storage(&bundle);
    assert_eq!(storage.account_uuid(), "profile-account");
    assert_eq!(storage.organization_uuid(), "profile-org");
    assert_eq!(storage.organization_name(), "Profile Org");
    assert_eq!(storage.device_ids(), bundle.device_ids());
}

#[tokio::test]
async fn candidate_exchange_survives_advisory_companion_failure() {
    let mut transport = CandidateTransport::successful();
    transport.inspect_failure = true;
    let auth = ClaudeAuth::with_clock(transport, FixedClock(SystemTime::UNIX_EPOCH));
    let bundle = auth
        .exchange_code_for_tokens(
            &SecretString::new("candidate-code").unwrap(),
            &SecretString::new("state").unwrap(),
            &candidate_pkce(),
        )
        .await
        .unwrap();
    assert_eq!(bundle.token_data().email(), "token@example.com");
    assert_eq!(bundle.user_info().account_uuid(), "token-account");
    assert_eq!(bundle.device_ids().len(), CLAUDE_DEVICE_POOL_SIZE);
}

#[tokio::test]
async fn candidate_refresh_uses_scope_preserves_rotation_and_fetches_profile() {
    let auth = ClaudeAuth::with_clock(
        CandidateTransport::successful(),
        FixedClock(SystemTime::UNIX_EPOCH),
    );
    let refreshed = auth
        .refresh_tokens(SecretString::new("old-refresh").unwrap())
        .await
        .unwrap();
    assert_eq!(refreshed.refresh_token().expose_secret(), "old-refresh");
    assert_eq!(refreshed.email(), "profile@example.com");
    assert_eq!(refreshed.account_uuid(), "profile-account");
    assert_eq!(refreshed.organization_uuid(), "profile-org");
    assert_eq!(refreshed.organization_name(), "Profile Org");
    let request = lock_recover(&auth.transport().refresh_requests)[0].clone();
    let request: serde_json::Value = serde_json::from_slice(&request).unwrap();
    assert_eq!(request["scope"], AUTH_SCOPE);
}

#[tokio::test]
async fn candidate_fetch_profile_rejects_missing_account_uuid() {
    let mut transport = CandidateTransport::successful();
    transport.profile_body = br#"{"account":{"email":"profile@example.com"}}"#.to_vec();
    let auth = ClaudeAuth::with_clock(transport, FixedClock(SystemTime::UNIX_EPOCH));
    assert!(auth
        .fetch_oauth_profile(&SecretString::new("candidate-access").unwrap())
        .await
        .is_err());
}

#[test]
fn candidate_storage_update_preserves_identity_when_refresh_omits_it() {
    let now = SystemTime::UNIX_EPOCH;
    let initial = ClaudeTokenData::new(
        SecretString::new("access").unwrap(),
        SecretString::new("refresh").unwrap(),
        "operator@example.com",
        now + Duration::from_secs(3600),
    )
    .with_identity("account", "org", "Organization");
    let mut storage = super::token::ClaudeTokenStorage::from_token_data(&initial, now, None);
    let refreshed = ClaudeTokenData::new(
        SecretString::new("new-access").unwrap(),
        SecretString::new("new-refresh").unwrap(),
        "",
        now + Duration::from_secs(7200),
    );
    storage.update_from_token_data(&refreshed, now + Duration::from_secs(10));
    assert_eq!(storage.account_uuid(), "account");
    assert_eq!(storage.organization_uuid(), "org");
    assert_eq!(storage.organization_name(), "Organization");
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
    ) -> Pin<Box<dyn Future<Output = Result<(), RefreshTransportFailure>> + Send + '_>> {
        Box::pin(async { Ok(()) })
    }
}

struct CapturingTransport {
    response: Mutex<Option<Result<RefreshHttpResponse, RefreshTransportFailure>>>,
    calls: AtomicUsize,
    timeouts: Mutex<Vec<Duration>>,
}

impl CapturingTransport {
    fn responding(response: RefreshHttpResponse) -> Self {
        Self {
            response: Mutex::new(Some(Ok(response))),
            calls: AtomicUsize::new(0),
            timeouts: Mutex::new(Vec::new()),
        }
    }
}

impl ClaudeRefreshTransport for CapturingTransport {
    fn execute<'a>(
        &'a self,
        _request: &'a RefreshRequest,
        timeout: Duration,
    ) -> Pin<
        Box<dyn Future<Output = Result<RefreshHttpResponse, RefreshTransportFailure>> + Send + 'a>,
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

// Disposition: adapted. Go asserts `http.Client.Timeout == 0`; Rust has no
// blanket client timeout and supplies a bounded timeout per OAuth operation.
#[cfg(feature = "anthropic-fingerprint-transport")]
#[tokio::test]
async fn anthropic_http_client_does_not_apply_a_hidden_blanket_timeout() {
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::TcpListener;

    use super::utls_transport::AnthropicHttpTransport;

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let endpoint = format!("http://{}/v1/oauth/token", listener.local_addr().unwrap());
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

    let transport = AnthropicHttpTransport::with_endpoint(&endpoint, None).unwrap();
    let request = RefreshRequest::new(SecretString::new("timeout-probe").unwrap());
    let started = tokio::time::Instant::now();
    let response = transport
        .execute(&request, Duration::from_millis(500))
        .await
        .unwrap();
    server.await.unwrap();

    assert!(started.elapsed() >= Duration::from_millis(50));
    assert!(format!("{response:?}").contains("status: 200"));
}

// Disposition: ported with the CTOX cancellation adaptation. Claude refresh is
// deliberately detached from caller cancellation and always receives its own
// exact 30-second operation timeout.
#[tokio::test]
async fn refresh_tokens_uses_independent_timeout() {
    let transport = CapturingTransport::responding(RefreshHttpResponse::new(
        400,
        None,
        None,
        br#"{"error":"probe"}"#.to_vec(),
    ));
    let auth = ClaudeAuth::with_clock(transport, FixedClock(SystemTime::UNIX_EPOCH));

    let error = auth
        .refresh_tokens(SecretString::new("independent-timeout-token").unwrap())
        .await
        .unwrap_err();
    assert_eq!(
        error,
        RefreshError::Http {
            status: 400,
            retryable: false
        }
    );
    assert_eq!(
        *lock_recover(&auth.transport().timeouts),
        vec![REFRESH_TIMEOUT]
    );
}

// Disposition: ported. The Rust coordinator fingerprints the refresh token
// rather than retaining it as the Go map key, but preserves the 429 cooldown.
#[tokio::test]
async fn refresh_tokens_with_retry_429_blocks_immediate_replay() {
    let now = SystemTime::UNIX_EPOCH + Duration::from_secs(1_000);
    let transport = CapturingTransport::responding(RefreshHttpResponse::new(
        429,
        Some("60".to_owned()),
        None,
        br#"{"error":"rate_limited"}"#.to_vec(),
    ));
    let auth = ClaudeAuth::with_clock(transport, FixedClock(now));
    let token = SecretString::new("dummy_refresh_token").unwrap();

    let first = auth
        .refresh_tokens_with_retry(token.clone(), 3)
        .await
        .unwrap_err();
    let second = auth.refresh_tokens_with_retry(token, 3).await.unwrap_err();

    assert_eq!(auth.transport().calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        first,
        RefreshError::RateLimited {
            blocked_until: now + Duration::from_secs(60)
        }
    );
    assert_eq!(first, second);
    assert!(format!("{first}").contains("temporarily blocked"));
}

struct BlockingTransport {
    calls: AtomicUsize,
    started: Semaphore,
    released: Semaphore,
}

impl BlockingTransport {
    fn new() -> Self {
        Self {
            calls: AtomicUsize::new(0),
            started: Semaphore::new(0),
            released: Semaphore::new(0),
        }
    }
}

impl ClaudeRefreshTransport for BlockingTransport {
    fn execute<'a>(
        &'a self,
        _request: &'a RefreshRequest,
        _timeout: Duration,
    ) -> Pin<
        Box<dyn Future<Output = Result<RefreshHttpResponse, RefreshTransportFailure>> + Send + 'a>,
    > {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.add_permits(1);
            self.released.acquire().await.unwrap().forget();
            Ok(RefreshHttpResponse::new(
                200,
                None,
                None,
                br#"{
                    "access_token":"new-access",
                    "refresh_token":"new-refresh",
                    "token_type":"Bearer",
                    "expires_in":3600,
                    "account":{"email_address":"shared@example.com"}
                }"#
                .to_vec(),
            ))
        })
    }
}

// Disposition: ported. All callers sharing one ClaudeAuth coordinator and one
// refresh-token fingerprint observe the same single upstream result.
#[tokio::test]
async fn refresh_tokens_deduplicates_concurrent_refresh() {
    let auth = Arc::new(ClaudeAuth::with_clock(
        BlockingTransport::new(),
        FixedClock(SystemTime::UNIX_EPOCH),
    ));
    let run = |auth: Arc<ClaudeAuth<BlockingTransport, FixedClock>>| {
        tokio::spawn(async move {
            auth.refresh_tokens(SecretString::new("shared-refresh-token").unwrap())
                .await
        })
    };

    let first = run(Arc::clone(&auth));
    auth.transport().started.acquire().await.unwrap().forget();
    let second = run(Arc::clone(&auth));
    tokio::time::sleep(Duration::from_millis(20)).await;
    assert_eq!(auth.transport().calls.load(Ordering::SeqCst), 1);
    auth.transport().released.add_permits(1);

    let first = first.await.unwrap().unwrap();
    let second = second.await.unwrap().unwrap();
    assert_eq!(first, second);
    assert_eq!(first.access_token().expose_secret(), "new-access");
    assert_eq!(auth.transport().calls.load(Ordering::SeqCst), 1);
}

// ref: internal/auth/xai/xai_auth_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime};

use base64::Engine as _;
use tokio::sync::Notify;

use super::*;
use crate::sdk::auth::LoginCancellation;

struct FixedClock(Mutex<SystemTime>);
impl XaiClock for FixedClock {
    fn now(&self) -> SystemTime {
        *self.0.lock().unwrap()
    }
    fn sleep<'a>(
        &'a self,
        duration: Duration,
        cancellation: &'a LoginCancellation,
    ) -> XaiSleepFuture<'a> {
        Box::pin(async move {
            if cancellation.is_cancelled() {
                return Err(XaiTransportFailure::Cancelled);
            }
            *self.0.lock().unwrap() += duration;
            Ok(())
        })
    }
}

#[derive(Default)]
struct SequenceTransport {
    responses: Mutex<VecDeque<XaiHttpResponse>>,
    requests: Mutex<Vec<RecordedRequest>>,
}

type RecordedRequest = (XaiHttpMethod, String, String, Option<String>);
impl SequenceTransport {
    fn with(responses: impl IntoIterator<Item = XaiHttpResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }
}
impl XaiHttpTransport for SequenceTransport {
    fn execute<'a>(
        &'a self,
        request: &'a XaiHttpRequest,
        _timeout: Duration,
        _cancellation: &'a LoginCancellation,
    ) -> XaiHttpFuture<'a> {
        Box::pin(async move {
            self.requests.lock().unwrap().push((
                request.method,
                request.url.clone(),
                String::from_utf8_lossy(&request.body).into_owned(),
                request.proxy_url.clone(),
            ));
            self.responses
                .lock()
                .unwrap()
                .pop_front()
                .ok_or(XaiTransportFailure::Protocol)
        })
    }
}

fn client(transport: Arc<dyn XaiHttpTransport>, clock: Arc<dyn XaiClock>) -> XaiAuth {
    XaiAuth::new(transport, clock, Arc::new(XaiRefreshCoordinator::default()))
}

#[test]
fn validates_only_https_xai_origins() {
    assert!(validate_oauth_endpoint("https://auth.x.ai/oauth2/token", "token_endpoint").is_ok());
    assert_eq!(
        validate_oauth_endpoint("http://auth.x.ai/token", "token_endpoint")
            .unwrap_err()
            .kind,
        XaiAuthErrorKind::InsecureEndpoint
    );
    assert_eq!(
        validate_oauth_endpoint("https://evil.example/token", "token_endpoint")
            .unwrap_err()
            .kind,
        XaiAuthErrorKind::ForeignEndpoint
    );
    assert_eq!(
        validate_oauth_endpoint("", "token_endpoint")
            .unwrap_err()
            .kind,
        XaiAuthErrorKind::MissingEndpoint
    );
}

#[tokio::test]
async fn request_device_code_posts_client_id_scope_and_explicit_proxy() {
    let transport = Arc::new(SequenceTransport::with([XaiHttpResponse::new(200, br#"{"device_code":"device-abc","user_code":"ABCD-1234","verification_uri":"https://accounts.x.ai/device","expires_in":1800,"interval":5}"#.to_vec())]));
    let auth = XaiAuth::with_proxy_url(
        transport.clone(),
        Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH))),
        Arc::new(XaiRefreshCoordinator::default()),
        Some(" socks5://proxy.test ".to_owned()),
    );
    let code = auth
        .request_device_code(
            &LoginCancellation::default(),
            "https://accounts.x.ai/device",
            "https://auth.x.ai/token",
        )
        .await
        .unwrap();
    assert_eq!(code.device_code, "device-abc");
    assert_eq!(code.token_endpoint, "https://auth.x.ai/token");
    let requests = transport.requests.lock().unwrap();
    assert!(requests[0].2.contains(&format!("client_id={CLIENT_ID}")));
    assert!(requests[0]
        .2
        .contains("scope=openid+profile+email+offline_access+grok-cli%3Aaccess+api%3Aaccess"));
    assert_eq!(requests[0].3.as_deref(), Some("socks5://proxy.test"));
}

#[tokio::test]
async fn poll_exchanges_device_code_and_extracts_identity() {
    let jwt = fake_jwt("user@x.ai", "sub-1");
    let transport = Arc::new(SequenceTransport::with([
        XaiHttpResponse::new(400, br#"{"error":"authorization_pending"}"#.to_vec()),
        XaiHttpResponse::new(200, format!(r#"{{"access_token":"access-1","refresh_token":"refresh-1","token_type":"Bearer","expires_in":3600,"id_token":"{jwt}"}}"#).into_bytes()),
    ]));
    let clock = Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH)));
    let auth = client(transport.clone(), clock.clone());
    let token = auth
        .poll_for_token(
            &LoginCancellation::default(),
            Some(&DeviceCodeResponse {
                device_code: "device-abc".into(),
                expires_in: 60,
                interval: 1,
                token_endpoint: "https://auth.x.ai/token".into(),
                ..DeviceCodeResponse::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(token.access_token().expose_secret(), "access-1");
    assert_eq!(token.refresh_token().unwrap().expose_secret(), "refresh-1");
    assert_eq!(token.email(), "user@x.ai");
    assert_eq!(token.subject(), "sub-1");
    assert_eq!(
        *clock.0.lock().unwrap(),
        SystemTime::UNIX_EPOCH + DEFAULT_POLL_INTERVAL
    );
    assert_eq!(transport.requests.lock().unwrap().len(), 2);
}

#[tokio::test]
async fn access_denied_is_a_typed_terminal_error() {
    let transport = Arc::new(SequenceTransport::with([XaiHttpResponse::new(
        400,
        br#"{"error":"access_denied","error_description":"rejected"}"#.to_vec(),
    )]));
    let auth = client(
        transport,
        Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH))),
    );
    let error = auth
        .poll_for_token(
            &LoginCancellation::default(),
            Some(&DeviceCodeResponse {
                device_code: "d".into(),
                expires_in: 60,
                token_endpoint: "https://auth.x.ai/token".into(),
                ..DeviceCodeResponse::default()
            }),
        )
        .await
        .unwrap_err();
    assert_eq!(error.kind, XaiAuthErrorKind::AccessDenied);
}

#[tokio::test]
async fn slow_down_adds_five_seconds_and_continues_polling() {
    let transport = Arc::new(SequenceTransport::with([
        XaiHttpResponse::new(400, br#"{"error":"slow_down"}"#.to_vec()),
        XaiHttpResponse::new(
            200,
            br#"{"access_token":"access-slow","expires_in":3600}"#.to_vec(),
        ),
    ]));
    let clock = Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH)));
    let auth = client(transport, clock.clone());
    let token = auth
        .poll_for_token(
            &LoginCancellation::default(),
            Some(&DeviceCodeResponse {
                device_code: "d".into(),
                expires_in: 60,
                interval: 5,
                token_endpoint: "https://auth.x.ai/token".into(),
                ..DeviceCodeResponse::default()
            }),
        )
        .await
        .unwrap();
    assert_eq!(token.access_token().expose_secret(), "access-slow");
    assert_eq!(
        *clock.0.lock().unwrap(),
        SystemTime::UNIX_EPOCH + Duration::from_secs(10)
    );
}

#[tokio::test]
async fn zero_expiry_stays_none_and_positive_expiry_uses_injected_clock() {
    let transport = Arc::new(SequenceTransport::with([
        XaiHttpResponse::new(200, br#"{"access_token":"a","expires_in":0}"#.to_vec()),
        XaiHttpResponse::new(200, br#"{"access_token":"b","expires_in":60}"#.to_vec()),
    ]));
    let auth = client(
        transport,
        Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH))),
    );
    let zero = auth
        .refresh_tokens(
            SecretString::new("r1").unwrap(),
            Some("https://auth.x.ai/token"),
        )
        .await
        .unwrap();
    let positive = auth
        .refresh_tokens(
            SecretString::new("r2").unwrap(),
            Some("https://auth.x.ai/token"),
        )
        .await
        .unwrap();
    assert_eq!(zero.expires_at(), None);
    assert_eq!(
        positive.expires_at(),
        Some(SystemTime::UNIX_EPOCH + Duration::from_secs(60))
    );
}

#[tokio::test]
async fn refresh_posts_client_id_and_refresh_token() {
    let transport = Arc::new(SequenceTransport::with([XaiHttpResponse::new(
        200,
        br#"{"access_token":"new-access","refresh_token":"new-refresh"}"#.to_vec(),
    )]));
    let auth = client(
        transport.clone(),
        Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH))),
    );
    let token = auth
        .refresh_tokens(
            SecretString::new("old-refresh").unwrap(),
            Some("https://auth.x.ai/token"),
        )
        .await
        .unwrap();
    assert_eq!(token.access_token().expose_secret(), "new-access");
    let body = &transport.requests.lock().unwrap()[0].2;
    assert!(body.contains("grant_type=refresh_token"));
    assert!(body.contains(&format!("client_id={CLIENT_ID}")));
    assert!(body.contains("refresh_token=old-refresh"));
}

struct BlockingRefreshTransport {
    calls: AtomicUsize,
    started: Notify,
    release: Notify,
}
impl XaiHttpTransport for BlockingRefreshTransport {
    fn execute<'a>(
        &'a self,
        _request: &'a XaiHttpRequest,
        _timeout: Duration,
        _cancellation: &'a LoginCancellation,
    ) -> XaiHttpFuture<'a> {
        Box::pin(async move {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            self.release.notified().await;
            Ok(XaiHttpResponse::new(
                200,
                br#"{"access_token":"new-access","refresh_token":"new-refresh"}"#.to_vec(),
            ))
        })
    }
}

#[tokio::test]
async fn concurrent_refresh_is_instance_coordinator_deduplicated() {
    let transport = Arc::new(BlockingRefreshTransport {
        calls: AtomicUsize::new(0),
        started: Notify::new(),
        release: Notify::new(),
    });
    let coordinator = Arc::new(XaiRefreshCoordinator::default());
    let clock: Arc<dyn XaiClock> = Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH)));
    let first = Arc::new(XaiAuth::new(
        transport.clone(),
        clock.clone(),
        coordinator.clone(),
    ));
    let second = Arc::new(XaiAuth::new(transport.clone(), clock, coordinator));
    let a = tokio::spawn(async move {
        first
            .refresh_tokens(
                SecretString::new("shared").unwrap(),
                Some("https://auth.x.ai/token"),
            )
            .await
    });
    transport.started.notified().await;
    let b = tokio::spawn(async move {
        second
            .refresh_tokens(
                SecretString::new("shared").unwrap(),
                Some("https://auth.x.ai/token"),
            )
            .await
    });
    tokio::task::yield_now().await;
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    transport.release.notify_one();
    for result in [a.await.unwrap(), b.await.unwrap()] {
        assert_eq!(result.unwrap().access_token().expose_secret(), "new-access");
    }
    assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
}

#[test]
fn storage_preserves_nil_vs_empty_without_plaintext_serialization() {
    let auth = client(
        Arc::new(SequenceTransport::default()),
        Arc::new(FixedClock(Mutex::new(SystemTime::UNIX_EPOCH))),
    );
    assert!(auth.create_token_storage(None).is_none());
    let token = TokenData::new(
        SecretString::new("access-secret-value").unwrap(),
        None,
        None,
        "",
        0,
        None,
        "",
        "",
    );
    let bundle = AuthBundle {
        token_data: token,
        last_refresh: SystemTime::UNIX_EPOCH,
        base_url: DEFAULT_API_BASE_URL.into(),
        redirect_uri: String::new(),
        token_endpoint: String::new(),
    };
    let storage = auth.create_token_storage(Some(&bundle)).unwrap();
    assert!(storage.credentials().refresh_token().is_none());
    assert!(storage.credentials().id_token().is_none());
    assert!(!format!("{storage:?}").contains("access-secret-value"));
}

fn fake_jwt(email: &str, subject: &str) -> String {
    let encode = |value: &[u8]| base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(value);
    format!(
        "{}.{}.sig",
        encode(br#"{"alg":"none"}"#),
        encode(format!(r#"{{"email":"{email}","sub":"{subject}"}}"#).as_bytes())
    )
}

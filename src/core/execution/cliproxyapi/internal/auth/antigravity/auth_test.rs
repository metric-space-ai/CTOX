// ref: internal/auth/antigravity/auth_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::sdk::auth::LoginCancellation;

use super::{
    AntigravityAuth, AntigravityAuthErrorKind, AntigravityFlowTransport, AntigravityHttpMethod,
    AntigravityHttpRequest, AntigravityHttpResponse, AntigravityHttpTransportFailure, SecretString,
};

#[derive(Debug)]
struct CapturedRequest {
    method: AntigravityHttpMethod,
    url: String,
    headers: std::collections::BTreeMap<String, String>,
    body: Vec<u8>,
}

struct ScriptTransport {
    responses: Mutex<VecDeque<AntigravityHttpResponse>>,
    requests: Mutex<Vec<CapturedRequest>>,
}

impl ScriptTransport {
    fn new(responses: impl IntoIterator<Item = AntigravityHttpResponse>) -> Self {
        Self {
            responses: Mutex::new(responses.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
        }
    }

    fn requests(&self) -> std::sync::MutexGuard<'_, Vec<CapturedRequest>> {
        self.requests.lock().unwrap()
    }
}

impl AntigravityFlowTransport for ScriptTransport {
    fn execute<'a>(
        &'a self,
        request: &'a AntigravityHttpRequest,
        timeout: Duration,
        _: &'a LoginCancellation,
    ) -> Pin<
        Box<
            dyn Future<Output = Result<AntigravityHttpResponse, AntigravityHttpTransportFailure>>
                + Send
                + 'a,
        >,
    > {
        Box::pin(async move {
            assert_eq!(timeout, Duration::from_secs(30));
            self.requests.lock().unwrap().push(CapturedRequest {
                method: request.method,
                url: request.url.clone(),
                headers: request.headers.clone(),
                body: request.body.to_vec(),
            });
            Ok(self.responses.lock().unwrap().pop_front().unwrap())
        })
    }
}

fn response(body: &str) -> AntigravityHttpResponse {
    AntigravityHttpResponse::new(200, body.as_bytes().to_vec())
}

fn auth_with(
    responses: impl IntoIterator<Item = AntigravityHttpResponse>,
) -> (AntigravityAuth, Arc<ScriptTransport>) {
    let transport = Arc::new(ScriptTransport::new(responses));
    let auth = AntigravityAuth::new(transport.clone());
    (auth, transport)
}

fn token() -> SecretString {
    SecretString::new("access-token").unwrap()
}

#[tokio::test]
async fn fetch_project_id_from_load_code_assist() {
    let (auth, transport) = auth_with([response(
        r#"{"cloudaicompanionProject":"cogent-snow-4mnnp"}"#,
    )]);
    let project = auth
        .fetch_project_id(&LoginCancellation::default(), &token())
        .await
        .unwrap();
    assert_eq!(project, "cogent-snow-4mnnp");

    let requests = transport.requests();
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    assert_eq!(request.method, AntigravityHttpMethod::Post);
    assert_eq!(
        request.url,
        "https://cloudcode-pa.googleapis.com/v1internal:loadCodeAssist"
    );
    assert_eq!(request.headers["Authorization"], "Bearer access-token");
    assert_eq!(request.headers["Accept"], "*/*");
    assert!(!request.headers.contains_key("X-Goog-Api-Client"));
    assert!(request.headers["User-Agent"].starts_with("antigravity/hub/"));
    assert!(!request.headers["User-Agent"].contains("google-api-nodejs-client/"));
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["metadata"]["ideType"], "ANTIGRAVITY");
}

#[tokio::test]
async fn fetch_project_id_falls_back_to_daily_onboard_user() {
    let (auth, transport) = auth_with([
        response(r#"{"allowedTiers":[{"id":"free-tier","isDefault":true}]}"#),
        response(
            r#"{"done":true,"response":{"cloudaicompanionProject":{"id":"cogent-snow-4mnnp","name":"cogent-snow-4mnnp","projectNumber":"22597072101"}}}"#,
        ),
    ]);
    let project = auth
        .fetch_project_id(&LoginCancellation::default(), &token())
        .await
        .unwrap();
    assert_eq!(project, "cogent-snow-4mnnp");

    let requests = transport.requests();
    assert_eq!(requests.len(), 2);
    let request = &requests[1];
    assert_eq!(
        request.url,
        "https://daily-cloudcode-pa.googleapis.com/v1internal:onboardUser"
    );
    assert_eq!(request.headers["Authorization"], "Bearer access-token");
    assert_eq!(request.headers["Accept"], "*/*");
    assert_eq!(request.headers["X-Goog-Api-Client"], "gl-node/22.21.1");
    assert!(request.headers["User-Agent"].starts_with("antigravity/hub/"));
    assert!(request.headers["User-Agent"].contains("google-api-nodejs-client/10.3.0"));
    let body: serde_json::Value = serde_json::from_slice(&request.body).unwrap();
    assert_eq!(body["tier_id"], "free-tier");
    assert_eq!(body["metadata"]["ide_type"], "ANTIGRAVITY");
    assert_eq!(body["metadata"]["ide_version"], "2.2.1");
}

#[tokio::test]
async fn current_tier_and_alternate_project_shapes_are_supported() {
    let (auth, transport) = auth_with([
        response(r#"{"currentTier":{"id":"paid-tier"}}"#),
        response(r#"{"done":true,"response":{"projectId":" project-42 "}}"#),
    ]);
    assert_eq!(
        auth.fetch_project_id(&LoginCancellation::default(), &token())
            .await
            .unwrap(),
        "project-42"
    );
    let requests = transport.requests();
    let body: serde_json::Value = serde_json::from_slice(&requests[1].body).unwrap();
    assert_eq!(body["tier_id"], "paid-tier");
}

#[tokio::test]
async fn token_exchange_uses_form_contract_and_redacts_secrets() {
    let (auth, transport) = auth_with([response(
        r#"{"access_token":"access-new","refresh_token":"refresh-new","expires_in":3600,"token_type":"Bearer"}"#,
    )]);
    let token_response = auth
        .exchange_code_for_tokens(
            &LoginCancellation::default(),
            "oauth-code",
            "http://localhost:51121/oauth-callback",
        )
        .await
        .unwrap();
    assert_eq!(token_response.access_token().expose_secret(), "access-new");
    assert_eq!(
        token_response.refresh_token().unwrap().expose_secret(),
        "refresh-new"
    );
    assert_eq!(token_response.expires_in, 3600);
    let rendered = format!("{token_response:?}");
    assert!(!rendered.contains("access-new"));
    assert!(!rendered.contains("refresh-new"));

    let requests = transport.requests();
    assert_eq!(requests[0].url, "https://oauth2.googleapis.com/token");
    let form: std::collections::HashMap<_, _> = url::form_urlencoded::parse(&requests[0].body)
        .into_owned()
        .collect();
    assert_eq!(form["code"], "oauth-code");
    assert_eq!(form["grant_type"], "authorization_code");
    assert_eq!(
        form["redirect_uri"],
        "http://localhost:51121/oauth-callback"
    );
}

#[tokio::test]
async fn fetch_user_info_requires_a_nonempty_email() {
    let (auth, transport) = auth_with([response(r#"{"email":" user@example.com "}"#)]);
    assert_eq!(
        auth.fetch_user_info(&LoginCancellation::default(), &token())
            .await
            .unwrap(),
        "user@example.com"
    );
    {
        let requests = transport.requests();
        assert_eq!(requests[0].method, AntigravityHttpMethod::Get);
        assert_eq!(
            requests[0].url,
            "https://www.googleapis.com/oauth2/v2/userinfo?alt=json"
        );
    }

    let (auth, _) = auth_with([response(r#"{"email":"  "}"#)]);
    assert_eq!(
        auth.fetch_user_info(&LoginCancellation::default(), &token())
            .await
            .unwrap_err()
            .kind,
        AntigravityAuthErrorKind::MissingEmail
    );
}

#[tokio::test]
async fn provider_errors_do_not_echo_response_or_request_secrets() {
    let secret_body = "provider oauth-code access-token must-not-leak";
    let (auth, _) = auth_with([AntigravityHttpResponse::new(
        403,
        secret_body.as_bytes().to_vec(),
    )]);
    let error = auth
        .fetch_user_info(&LoginCancellation::default(), &token())
        .await
        .unwrap_err();
    assert_eq!(error.kind, AntigravityAuthErrorKind::UserInfo);
    assert_eq!(error.status, Some(403));
    let rendered = format!("{error:?} {error}");
    assert!(!rendered.contains(secret_body));
    assert!(!rendered.contains("access-token"));
}

#[tokio::test]
async fn cancellation_fences_onboarding_before_transport() {
    let (auth, transport) = auth_with([]);
    let cancellation = LoginCancellation::default();
    cancellation.cancel();
    let error = auth
        .onboard_user(&cancellation, &token(), "free-tier")
        .await
        .unwrap_err();
    assert_eq!(error.kind, AntigravityAuthErrorKind::Cancelled);
    assert!(transport.requests().is_empty());
}

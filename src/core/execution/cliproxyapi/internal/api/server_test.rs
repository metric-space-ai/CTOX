// ref: internal/api/server_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::{Arc, Mutex};

use crate::internal::config::{ProviderCompatConfig, SdkConfig};

use super::server_options::{ServerOptions, ServerReloadHook};
use super::server_reload::{ServerConfigReloader, ServerConfigSnapshot};
use super::server_routes::{
    is_business_data_route, resolve_server_route, CodexAlphaSearchClient, CodexAlphaSearchError,
    CodexAlphaSearchFuture, CodexAlphaSearchRefresher, CodexAlphaSearchResponse,
    CodexAlphaSearchTransport, CodexAlphaSearchTransportRequest, ServerRoute,
};
use crate::sdk::cliproxy::auth::Auth;
use crate::sdk::cliproxy::executor::Headers;

#[derive(Default)]
struct Hook(Mutex<Vec<u64>>);
impl ServerReloadHook for Hook {
    fn config_reloaded(&self, revision: u64) {
        self.0.lock().unwrap().push(revision);
    }
}

#[test]
fn public_provider_and_control_plane_routes_are_explicit() {
    assert_eq!(resolve_server_route("/healthz"), ServerRoute::Health);
    assert_eq!(
        resolve_server_route("/v1/models?provider=claude"),
        ServerRoute::Models
    );
    assert_eq!(
        resolve_server_route("/v1/responses"),
        ServerRoute::Responses
    );
    assert_eq!(resolve_server_route("/v1/messages"), ServerRoute::Messages);
    assert_eq!(
        resolve_server_route("/v1/chat/completions"),
        ServerRoute::ChatCompletions
    );
    assert_eq!(
        resolve_server_route("/backend-api/codex/responses/compact"),
        ServerRoute::ResponsesCompact
    );
    assert_eq!(
        resolve_server_route("/v1beta/models/gemini-pro:generateContent"),
        ServerRoute::Gemini
    );
    assert_eq!(
        resolve_server_route("/codex/callback?code=redacted"),
        ServerRoute::OAuthCallback
    );
    assert_eq!(
        resolve_server_route("/v0/management/usage"),
        ServerRoute::Management
    );
    assert_eq!(resolve_server_route("/interactions"), ServerRoute::NotFound);
    assert!(ServerRoute::Responses.allows_method("POST"));
    assert!(ServerRoute::Responses.allows_method("GET"));
    assert!(!ServerRoute::Responses.allows_method("PUT"));
}

#[test]
fn browser_business_data_is_never_routed_through_http() {
    assert!(is_business_data_route("/business-os/data/modules"));
    assert!(is_business_data_route("/rxdb/collections"));
    assert_eq!(
        resolve_server_route("/business-os/data/modules"),
        ServerRoute::NotFound
    );
    assert_eq!(
        resolve_server_route("/rxdb/collections"),
        ServerRoute::NotFound
    );
}

#[test]
fn reload_snapshot_is_monotonic_atomic_and_hook_receives_only_revision() {
    let hook = Arc::new(Hook::default());
    let reloader = ServerConfigReloader::new(
        ServerConfigSnapshot {
            revision: 1,
            providers: ProviderCompatConfig::default(),
            sdk: SdkConfig::default(),
        },
        Some(hook.clone()),
    );
    let old = reloader.snapshot();
    assert!(!reloader.publish(ServerConfigSnapshot {
        revision: 1,
        providers: ProviderCompatConfig::default(),
        sdk: SdkConfig::default()
    }));
    let mut providers = ProviderCompatConfig::default();
    providers.codex.optimize_multi_agent_v2 = true;
    assert!(reloader.publish(ServerConfigSnapshot {
        revision: 2,
        providers,
        sdk: SdkConfig::default()
    }));
    assert_eq!(old.revision, 1);
    assert_eq!(reloader.snapshot().revision, 2);
    assert!(reloader.snapshot().sdk.codex_optimize_multi_agent_v2);
    assert_eq!(*hook.0.lock().unwrap(), [2]);
}

#[test]
fn options_debug_reports_capabilities_not_authority_objects() {
    let options = ServerOptions {
        local_management_enabled: true,
        ..ServerOptions::default()
    };
    let debug = format!("{options:?}");
    assert!(debug.contains("local_management_enabled"));
    assert!(!debug.contains("password"));
}

#[derive(Default)]
struct AlphaSearchProbe(Mutex<Vec<CodexAlphaSearchTransportRequest>>);

impl CodexAlphaSearchTransport for AlphaSearchProbe {
    fn execute<'a>(
        &'a self,
        request: CodexAlphaSearchTransportRequest,
    ) -> CodexAlphaSearchFuture<'a> {
        let status = if request.auth_id == "fresh" { 200 } else { 401 };
        self.0.lock().unwrap().push(request);
        Box::pin(async move {
            Ok(CodexAlphaSearchResponse {
                status,
                headers: Headers::new(),
                body: b"result".to_vec(),
            })
        })
    }
}

struct AlwaysUnauthorizedAlphaSearch;

impl CodexAlphaSearchTransport for AlwaysUnauthorizedAlphaSearch {
    fn execute<'a>(&'a self, _: CodexAlphaSearchTransportRequest) -> CodexAlphaSearchFuture<'a> {
        Box::pin(async {
            Ok(CodexAlphaSearchResponse {
                status: 401,
                headers: Headers::new(),
                body: Vec::new(),
            })
        })
    }
}

#[derive(Default)]
struct AlphaSearchRefresh(Mutex<usize>);

impl CodexAlphaSearchRefresher for AlphaSearchRefresh {
    fn report_unauthorized<'a>(
        &'a self,
        _: &'a Auth,
        _: &'a str,
    ) -> super::server_routes::CodexAlphaSearchStatusFuture<'a> {
        *self.0.lock().unwrap() += 1;
        Box::pin(async { Ok(()) })
    }

    fn refresh_after_unauthorized<'a>(
        &'a self,
        current: &'a Auth,
        _: &'a str,
    ) -> super::server_routes::CodexAlphaSearchSelectionFuture<'a> {
        let mut refreshed = current.clone();
        refreshed.id = "fresh".to_owned();
        refreshed
            .metadata
            .insert("account_id".to_owned(), serde_json::json!("account-fresh"));
        Box::pin(async move { Ok(refreshed) })
    }
}

#[tokio::test]
async fn alpha_search_refreshes_home_unauthorized_once_and_rebuilds_request() {
    let transport = Arc::new(AlphaSearchProbe::default());
    let refresher = Arc::new(AlphaSearchRefresh::default());
    let client = CodexAlphaSearchClient::new(transport.clone()).with_refresher(refresher.clone());
    let mut auth = Auth::default();
    auth.id = "stale".to_owned();
    auth.metadata
        .insert("access_token".to_owned(), serde_json::json!("stale-token"));
    let response = client
        .execute(
            &auth,
            "gpt-live",
            &Headers::from_iter([("User-Agent".to_owned(), vec!["codex/1".to_owned()])]),
            br#"{"query":"rust"}"#,
        )
        .await
        .expect("retry");
    assert_eq!(response.status, 200);
    let requests = transport.0.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(requests[0].auth_id, "stale");
    assert_eq!(requests[1].auth_id, "fresh");
    assert_eq!(
        requests[1].headers.get("Chatgpt-Account-Id"),
        Some(&vec!["account-fresh".to_owned()])
    );
    assert_eq!(*refresher.0.lock().unwrap(), 1);
}

#[tokio::test]
async fn alpha_search_reports_second_unauthorized_after_retry() {
    let refresher = Arc::new(AlphaSearchRefresh::default());
    let client = CodexAlphaSearchClient::new(Arc::new(AlwaysUnauthorizedAlphaSearch))
        .with_refresher(refresher.clone());
    let mut auth = Auth::default();
    auth.id = "stale".to_owned();
    let response = client
        .execute(&auth, "gpt-live", &Headers::new(), b"{}")
        .await
        .expect("401 response is forwarded");
    assert_eq!(response.status, 401);
    assert_eq!(*refresher.0.lock().unwrap(), 2);
}

#[tokio::test]
async fn alpha_search_api_key_requires_explicit_base_url() {
    let transport = Arc::new(AlphaSearchProbe::default());
    let client = CodexAlphaSearchClient::new(transport);
    let mut auth = Auth::default();
    auth.id = "key".to_owned();
    auth.attributes
        .insert("auth_kind".to_owned(), "api_key".to_owned());
    assert_eq!(
        client.execute(&auth, "model", &Headers::new(), b"{}").await,
        Err(CodexAlphaSearchError::MissingApiKeyBaseUrl)
    );
}

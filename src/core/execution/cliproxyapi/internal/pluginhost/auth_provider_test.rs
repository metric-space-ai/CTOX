// ref: internal/pluginhost/auth_provider_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: auth calls use process RPC and typed host configuration
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use serde_json::{value::to_raw_value, Value};
use tokio::sync::mpsc;

use crate::sdk::pluginabi::{
    Envelope, METHOD_AUTH_LOGIN_POLL, METHOD_AUTH_LOGIN_START, METHOD_AUTH_PARSE,
    METHOD_AUTH_REFRESH,
};
use crate::sdk::pluginapi::{
    AuthLoginPollRequest, AuthLoginStartRequest, AuthParseRequest, AuthProvider,
    AuthRefreshRequest, HostConfigSummary,
};

use super::abi::{PluginCall, PluginClient, PluginFuture, PluginStream};
use super::adapters::RpcCapabilityClient;
use super::auth_provider::{HostConfigSummarySource, RpcAuthProvider};
use super::callback_contexts::CallbackContextRegistry;
use super::rpc_client::RpcPluginClient;

struct Client {
    calls: Mutex<Vec<PluginCall>>,
    responses: Mutex<VecDeque<Value>>,
}

impl PluginClient for Client {
    fn call<'a>(&'a self, call: PluginCall) -> PluginFuture<'a, Envelope> {
        Box::pin(async move {
            self.calls
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .push(call);
            let value = self
                .responses
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .pop_front()
                .unwrap();
            Ok(Envelope::success(Some(to_raw_value(&value).unwrap())))
        })
    }

    fn call_stream<'a>(&'a self, _call: PluginCall) -> PluginFuture<'a, PluginStream> {
        Box::pin(async {
            let (_sender, receiver) = mpsc::channel(1);
            Ok(PluginStream { chunks: receiver })
        })
    }

    fn shutdown<'a>(&'a self) -> PluginFuture<'a, ()> {
        Box::pin(async { Ok(()) })
    }
}

struct Host;
impl HostConfigSummarySource for Host {
    fn snapshot(&self) -> HostConfigSummary {
        HostConfigSummary {
            auth_dir: "/typed/auth".to_owned(),
            force_model_prefix: true,
            ..HostConfigSummary::default()
        }
    }
}

fn configured(responses: Vec<Value>) -> (RpcAuthProvider, Arc<Client>) {
    let client = Arc::new(Client {
        calls: Mutex::new(Vec::new()),
        responses: Mutex::new(responses.into()),
    });
    let provider = RpcAuthProvider::new(
        " Claude ",
        RpcCapabilityClient::new(
            "plugin-a",
            RpcPluginClient::new(client.clone()),
            CallbackContextRegistry::new(),
        )
        .unwrap(),
        Arc::new(Host),
    )
    .unwrap();
    (provider, client)
}

#[tokio::test]
async fn auth_provider_forwards_all_methods_with_normalized_identity_and_host() {
    let (provider, client) = configured(vec![
        serde_json::json!({"Handled": false}),
        serde_json::json!({"Provider": "claude", "URL": "https://login", "State": "s"}),
        serde_json::json!({"Status": "pending"}),
        serde_json::json!({"Auth": {"Provider": "claude"}}),
    ]);
    assert_eq!(provider.identifier(), "claude");
    provider
        .parse_auth(AuthParseRequest::default())
        .await
        .unwrap();
    provider
        .start_login(AuthLoginStartRequest {
            base_url: " https://local.invalid ".to_owned(),
            ..AuthLoginStartRequest::default()
        })
        .await
        .unwrap();
    provider
        .poll_login(AuthLoginPollRequest {
            state: " state ".to_owned(),
            ..AuthLoginPollRequest::default()
        })
        .await
        .unwrap();
    provider
        .refresh_auth(AuthRefreshRequest::default())
        .await
        .unwrap();

    let calls = client
        .calls
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner);
    assert_eq!(
        calls
            .iter()
            .map(|call| call.method.as_str())
            .collect::<Vec<_>>(),
        [
            METHOD_AUTH_PARSE,
            METHOD_AUTH_LOGIN_START,
            METHOD_AUTH_LOGIN_POLL,
            METHOD_AUTH_REFRESH
        ]
    );
    for call in calls.iter() {
        let value: Value = serde_json::from_str(call.payload.get()).unwrap();
        assert_eq!(
            value
                .get("Provider")
                .or_else(|| value.get("AuthProvider"))
                .unwrap(),
            "claude"
        );
        assert_eq!(value["Host"]["AuthDir"], "/typed/auth");
    }
    let login: Value = serde_json::from_str(calls[1].payload.get()).unwrap();
    assert_eq!(login["BaseUrl"], "https://local.invalid");
    let poll: Value = serde_json::from_str(calls[2].payload.get()).unwrap();
    assert_eq!(poll["State"], "state");
}

#[test]
fn auth_provider_rejects_empty_identifier() {
    let client = Arc::new(Client {
        calls: Mutex::new(Vec::new()),
        responses: Mutex::new(VecDeque::new()),
    });
    assert!(RpcAuthProvider::new(
        "  ",
        RpcCapabilityClient::new(
            "plugin-a",
            RpcPluginClient::new(client),
            CallbackContextRegistry::new(),
        )
        .unwrap(),
        Arc::new(Host),
    )
    .is_err());
}

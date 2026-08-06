// Origin: CTOX supplemental tests for sdk/cliproxy/types.go.
// License: AGPL-3.0-only

use std::sync::{Arc, Mutex};

use super::providers::LoadContext;
use super::types::*;

struct Parser;

impl PluginAuthParser for Parser {
    fn parse_auth<'a>(
        &'a self,
        _request: PluginAuthParseRequest,
    ) -> PluginAuthParseFuture<'a, Option<super::auth::Auth>> {
        Box::pin(async { Ok((None, false)) })
    }
}

fn validated_config() -> crate::internal::config::ValidatedRuntimeConfig {
    let config: crate::internal::config::CliproxyRuntimeConfig =
        serde_json::from_value(serde_json::json!({
            "claude_accounts": [{
                "id": "claude-a",
                "access_token_secret": {"scope": "cliproxy", "name": "access"},
                "refresh_token_secret": {"scope": "cliproxy", "name": "refresh"}
            }]
        }))
        .unwrap();
    config.validate().unwrap()
}

#[tokio::test]
async fn empty_watcher_preserves_upstream_nil_no_op_results() {
    let watcher = WatcherWrapper::default();
    assert_eq!(watcher.start(&LoadContext::default()).await, Ok(()));
    assert_eq!(watcher.stop(), Ok(()));
    assert!(!watcher.reload_config_if_changed());
    assert!(!watcher.dispatch_runtime_auth_update(AuthUpdate::default()));
    assert!(!watcher.dispatch_persisted_auth_update(AuthUpdate::default()));
    assert!(watcher.snapshot_auths().is_empty());
}

#[tokio::test]
async fn watcher_forwards_each_bound_capability_and_reports_reload_presence() {
    let calls = Arc::new(Mutex::new(Vec::new()));
    let start_calls = calls.clone();
    let stop_calls = calls.clone();
    let reload_calls = calls.clone();
    let runtime_calls = calls.clone();
    let persisted_calls = calls.clone();
    let snapshot_calls = calls.clone();
    let config_calls = calls.clone();
    let queue_calls = calls.clone();
    let parser_calls = calls.clone();
    let watcher = WatcherWrapper::new(WatcherBindings {
        start: Some(Arc::new(move |_| {
            start_calls.lock().unwrap().push("start");
            Box::pin(async { Ok(()) })
        })),
        stop: Some(Arc::new(move || {
            stop_calls.lock().unwrap().push("stop");
            Ok(())
        })),
        reload_config_if_changed: Some(Arc::new(move || {
            reload_calls.lock().unwrap().push("reload")
        })),
        dispatch_runtime_update: Some(Arc::new(move |_| {
            runtime_calls.lock().unwrap().push("runtime");
            true
        })),
        dispatch_persisted_auth: Some(Arc::new(move |_| {
            persisted_calls.lock().unwrap().push("persisted");
            true
        })),
        snapshot_auths: Some(Arc::new(move || {
            snapshot_calls.lock().unwrap().push("snapshot");
            Vec::new()
        })),
        set_config: Some(Arc::new(move |_| {
            config_calls.lock().unwrap().push("config")
        })),
        set_update_queue: Some(Arc::new(move |_| queue_calls.lock().unwrap().push("queue"))),
        set_plugin_auth_parser: Some(Arc::new(move |_| {
            parser_calls.lock().unwrap().push("parser")
        })),
    });

    watcher.start(&LoadContext::default()).await.unwrap();
    watcher.stop().unwrap();
    assert!(watcher.reload_config_if_changed());
    assert!(watcher.dispatch_runtime_auth_update(AuthUpdate::default()));
    assert!(watcher.dispatch_persisted_auth_update(AuthUpdate::default()));
    assert!(watcher.snapshot_auths().is_empty());
    watcher.set_config(validated_config());
    let (sender, _receiver) = tokio::sync::mpsc::channel(1);
    watcher.set_auth_update_queue(sender);
    watcher.set_plugin_auth_parser(Arc::new(Parser));
    assert_eq!(
        *calls.lock().unwrap(),
        [
            "start",
            "stop",
            "reload",
            "runtime",
            "persisted",
            "snapshot",
            "config",
            "queue",
            "parser"
        ]
    );
}

#[test]
fn watcher_debug_reports_capabilities_without_callback_details() {
    let watcher = WatcherWrapper::new(WatcherBindings {
        stop: Some(Arc::new(|| Err(WatcherError::Stop))),
        ..WatcherBindings::default()
    });
    let debug = format!("{watcher:?}");
    assert!(debug.contains("can_stop: true"));
    assert_eq!(watcher.stop(), Err(WatcherError::Stop));
}

#[tokio::test]
async fn token_provider_trait_uses_existing_stateless_provider_capability() {
    let provider = super::providers::new_file_token_client_provider();
    let result = TokenClientProvider::load(&provider, &LoadContext::default(), &validated_config())
        .await
        .unwrap();
    assert_eq!(result.successful_authed, 0);
}

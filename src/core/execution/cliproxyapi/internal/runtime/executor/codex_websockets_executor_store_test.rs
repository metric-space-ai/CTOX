// ref: internal/runtime/executor/codex_websockets_executor_store_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::codex_websockets_session::CodexWebsocketSessionStore;

#[tokio::test]
async fn close_all_releases_executor_owned_sessions() {
    let store = CodexWebsocketSessionStore::default();
    store.get_or_create("one").unwrap();
    store.get_or_create("two").unwrap();
    store.close_all().await;
    assert!(store.is_empty());
}

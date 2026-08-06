// ref: internal/pluginhost/stream_bridge_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: bounded process-stream lifecycle and cancellation parity
// License: MIT (upstream); modifications AGPL-3.0-only

use super::stream_bridge::{StreamBridge, StreamBridgeError};

#[tokio::test]
async fn emit_preserves_order_and_close_is_terminal() {
    let bridge = StreamBridge::new();
    let (id, mut chunks, lease) = bridge.open("plugin-a");
    bridge.emit("plugin-a", &id, b"one".to_vec()).await.unwrap();
    bridge.emit("plugin-a", &id, b"two".to_vec()).await.unwrap();
    bridge.close("plugin-a", &id, None).await.unwrap();
    assert_eq!(chunks.recv().await.unwrap().payload, b"one");
    assert_eq!(chunks.recv().await.unwrap().payload, b"two");
    assert!(chunks.recv().await.is_none());
    assert!(lease.is_cancelled());
    assert_eq!(
        bridge.emit("plugin-a", &id, b"late".to_vec()).await,
        Err(StreamBridgeError::NotOpen)
    );
}

#[tokio::test]
async fn plugin_error_is_delivered_as_final_chunk_without_payload() {
    let bridge = StreamBridge::new();
    let (id, mut chunks, _lease) = bridge.open("plugin-a");
    bridge
        .close("plugin-a", &id, Some("provider failed".to_owned()))
        .await
        .unwrap();
    let terminal = chunks.recv().await.unwrap();
    assert!(terminal.payload.is_empty());
    assert_eq!(terminal.error.unwrap().to_string(), "provider failed");
    assert!(chunks.recv().await.is_none());
}

#[tokio::test]
async fn dropping_lease_aborts_registry_and_receiver() {
    let bridge = StreamBridge::new();
    let (id, mut chunks, lease) = bridge.open("plugin-a");
    assert!(bridge.is_open(&id));
    drop(lease);
    assert!(!bridge.is_open(&id));
    assert!(chunks.recv().await.is_none());
}

#[tokio::test]
async fn stream_operations_are_owner_bound() {
    let bridge = StreamBridge::new();
    let (id, _chunks, _lease) = bridge.open("plugin-a");
    assert_eq!(
        bridge.emit("plugin-b", &id, b"wrong".to_vec()).await,
        Err(StreamBridgeError::WrongOwner)
    );
    assert_eq!(
        bridge.close("plugin-b", &id, None).await,
        Err(StreamBridgeError::WrongOwner)
    );
    bridge.close("plugin-a", &id, None).await.unwrap();
}

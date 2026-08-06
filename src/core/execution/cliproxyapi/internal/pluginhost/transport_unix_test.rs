// Origin: CTOX
// Port-Status: adapted_to_ctox
// Port-Note: Unix LocalTransport isolation evidence
// License: AGPL-3.0-only

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use base64::Engine as _;
use tokio::net::UnixStream;

use super::process_transport::{read_process_message, write_process_message};
use super::rpc_schema::{decode_upstream_json, ProcessMessage};
use super::transport_unix::*;
use crate::sdk::pluginabi::SCHEMA_VERSION;

fn test_root(label: &str) -> PathBuf {
    PathBuf::from("/tmp").join(format!("ctox-plugin-{label}-{}", uuid::Uuid::new_v4()))
}

#[tokio::test]
async fn real_unix_socket_enforces_permissions_and_handshake() {
    const TOKEN: &[u8] = b"0123456789abcdef0123456789abcdef";
    let root = test_root("handshake");
    fs::create_dir(&root).unwrap();
    let endpoint = UnixPluginEndpoint::bind(&root, "plugin-one").unwrap();
    let socket_path = endpoint.socket_path().to_owned();
    assert_eq!(
        fs::metadata(socket_path.parent().unwrap())
            .unwrap()
            .permissions()
            .mode()
            & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(&socket_path).unwrap().permissions().mode() & 0o777,
        0o600
    );

    let peer = tokio::spawn(async move {
        let mut stream = UnixStream::connect(socket_path).await.unwrap();
        let request = read_process_message(&mut stream).await.unwrap().unwrap();
        let ProcessMessage::Request {
            request_id,
            method,
            payload,
            ..
        } = request
        else {
            panic!("expected handshake request");
        };
        assert_eq!(method, "ctox.handshake");
        let request: HandshakeRequest = decode_upstream_json(&payload).unwrap();
        let nonce = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(request.nonce)
            .unwrap();
        let response = handshake_response_message(
            request_id,
            "plugin-one".into(),
            request.schema_version,
            handshake_proof(TOKEN, &nonce, "plugin-one", request.schema_version),
        )
        .unwrap();
        write_process_message(&mut stream, &response).await.unwrap();
    });
    let connection = endpoint.accept_verified("plugin-one", TOKEN).await.unwrap();
    assert_eq!(connection.plugin_id(), "plugin-one");
    peer.await.unwrap();
    drop(connection);
    drop(endpoint);
    assert!(!root.join(".cpa/plugin-one/s").exists());
    fs::remove_dir_all(&root).unwrap();
}

#[tokio::test]
async fn claimed_plugin_id_mismatch_fails_closed() {
    const TOKEN: &[u8] = b"0123456789abcdef0123456789abcdef";
    let root = test_root("mismatch");
    fs::create_dir(&root).unwrap();
    let endpoint = UnixPluginEndpoint::bind(&root, "expected").unwrap();
    let socket_path = endpoint.socket_path().to_owned();
    let peer = tokio::spawn(async move {
        let mut stream = UnixStream::connect(socket_path).await.unwrap();
        let request = read_process_message(&mut stream).await.unwrap().unwrap();
        let ProcessMessage::Request {
            request_id,
            payload,
            ..
        } = request
        else {
            panic!("expected handshake request");
        };
        let request: HandshakeRequest = decode_upstream_json(&payload).unwrap();
        let nonce = base64::engine::general_purpose::URL_SAFE_NO_PAD
            .decode(request.nonce)
            .unwrap();
        let response = handshake_response_message(
            request_id,
            "imposter".into(),
            SCHEMA_VERSION,
            handshake_proof(TOKEN, &nonce, "imposter", SCHEMA_VERSION),
        )
        .unwrap();
        write_process_message(&mut stream, &response).await.unwrap();
    });
    assert!(matches!(
        endpoint.accept_verified("expected", TOKEN).await,
        Err(UnixTransportError::Handshake)
    ));
    peer.await.unwrap();
    drop(endpoint);
    fs::remove_dir_all(&root).unwrap();
}

#[test]
fn stale_regular_file_and_symlink_are_never_deleted() {
    let root = test_root("unsafe");
    let instance = root.join(".cpa/plugin-one");
    fs::create_dir_all(&instance).unwrap();
    let socket = instance.join("s");
    fs::write(&socket, b"user-owned").unwrap();
    assert!(matches!(
        UnixPluginEndpoint::bind(&root, "plugin-one"),
        Err(UnixTransportError::UnsafePath)
    ));
    assert_eq!(fs::read(&socket).unwrap(), b"user-owned");
    fs::remove_dir_all(&root).unwrap();

    let root = test_root("symlink");
    fs::create_dir(&root).unwrap();
    let target = root.join("target");
    fs::create_dir(&target).unwrap();
    std::os::unix::fs::symlink(&target, root.join(".cpa")).unwrap();
    assert!(matches!(
        UnixPluginEndpoint::bind(&root, "plugin-one"),
        Err(UnixTransportError::UnsafePath)
    ));
    fs::remove_dir_all(&root).unwrap();
}

#[tokio::test]
async fn drop_never_removes_a_replacement_socket() {
    let root = test_root("replacement");
    fs::create_dir(&root).unwrap();
    let endpoint = UnixPluginEndpoint::bind(&root, "plugin-one").unwrap();
    let socket = endpoint.socket_path().to_owned();
    fs::remove_file(&socket).unwrap();
    let replacement = std::os::unix::net::UnixListener::bind(&socket).unwrap();
    drop(endpoint);
    assert!(socket.exists());
    drop(replacement);
    fs::remove_file(&socket).unwrap();
    fs::remove_dir_all(&root).unwrap();
}

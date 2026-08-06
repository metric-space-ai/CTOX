// ref: internal/pluginhost/loader_windows_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: Windows DLL shadow loading is replaced by named-pipe child processes
// License: MIT (upstream); modifications AGPL-3.0-only

use std::path::Path;

use super::platform::{plugin_file_from_name, PluginPlatform};

#[test]
fn windows_discovers_process_artifacts_not_dynamic_libraries() {
    let platform = PluginPlatform::process("windows", "aarch64");
    assert_eq!(platform.executable_suffix, ".ctox-plugin.exe");
    let file = plugin_file_from_name(
        Path::new(r"C:\typed\plugins"),
        "alpha-v1.2.3.ctox-plugin.exe",
        &platform.executable_suffix,
    )
    .unwrap();
    assert_eq!(file.id, "alpha");
    assert_eq!(file.version.as_deref(), Some("1.2.3"));
    assert!(plugin_file_from_name(
        Path::new(r"C:\typed\plugins"),
        "alpha.dll",
        &platform.executable_suffix,
    )
    .is_none());
}

#[cfg(windows)]
#[test]
fn windows_named_pipe_handshake_proof_is_scoped_and_redacted() {
    use super::transport_windows::{handshake_proof, HandshakeResponse};

    let token = [7_u8; 32];
    let proof = handshake_proof(&token, b"nonce", "alpha", 2);
    assert_eq!(proof, handshake_proof(&token, b"nonce", "alpha", 2));
    assert_ne!(proof, handshake_proof(&token, b"nonce-2", "alpha", 2));
    let response = HandshakeResponse {
        schema_version: 2,
        plugin_id: "alpha".to_owned(),
        proof: proof.clone(),
    };
    let debug = format!("{response:?}");
    assert!(!debug.contains(&proof));
    assert!(debug.contains("[REDACTED]"));
}

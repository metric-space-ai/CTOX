// ref: internal/runtime/executor/helps/utls_client_resumption_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: validates the contract consumed by the host-owned transport factory.
// License: MIT (upstream); modifications AGPL-3.0-only

use super::utls_client::{
    claude_code_transport_scope_key, CLAUDE_CODE_OMIT_EMPTY_PSK,
    CLAUDE_CODE_SESSION_CACHE_CAPACITY, CLAUDE_CODE_SKIP_RESUMPTION_WITHOUT_PSK_EXTENSION,
    CLAUDE_CODE_TLS_EXTENSIONS,
};

#[test]
fn resumption_profile_is_bounded_psk_last_and_proxy_scoped() {
    assert_eq!(CLAUDE_CODE_SESSION_CACHE_CAPACITY, 32);
    let padding = CLAUDE_CODE_TLS_EXTENSIONS
        .iter()
        .position(|extension| *extension == "boring_padding")
        .unwrap();
    let psk = CLAUDE_CODE_TLS_EXTENSIONS
        .iter()
        .position(|extension| *extension == "pre_shared_key")
        .unwrap();
    assert_eq!(psk, padding + 1);
    assert!(std::hint::black_box(CLAUDE_CODE_OMIT_EMPTY_PSK));
    assert!(std::hint::black_box(
        CLAUDE_CODE_SKIP_RESUMPTION_WITHOUT_PSK_EXTENSION
    ));
    assert_eq!(
        claude_code_transport_scope_key(None),
        claude_code_transport_scope_key(Some(""))
    );
    assert_ne!(
        claude_code_transport_scope_key(Some("http://proxy-a")),
        claude_code_transport_scope_key(Some("http://proxy-b"))
    );
}

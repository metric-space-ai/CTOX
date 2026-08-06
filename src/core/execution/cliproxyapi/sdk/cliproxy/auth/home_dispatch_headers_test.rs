// ref: sdk/cliproxy/auth/home_dispatch_headers_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: verifies normalized dispatch headers through the injected transport boundary
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;

use crate::internal::home::{Client, HomeConfig, HomeTransport};

use super::home_execution_paths_test::TestHomeTransport;

#[test]
fn dispatch_headers_are_normalized_without_global_request_state() {
    let transport = TestHomeTransport::with_auth_ids(&["auth"]);
    let facade: std::sync::Arc<dyn HomeTransport> = transport.clone();
    let client = Client::new(
        HomeConfig {
            enabled: true,
            ..HomeConfig::default()
        },
        facade,
    );
    client
        .rpop_auth(
            "model",
            "session",
            BTreeMap::from([
                ("Authorization".into(), "Bearer token".into()),
                ("X-Tenant".into(), "one".into()),
            ]),
            1,
            "strict",
        )
        .unwrap();
    let payload: serde_json::Value = serde_json::from_slice(&transport.requests()[0]).unwrap();
    assert_eq!(payload["headers"]["authorization"], "Bearer token");
    assert_eq!(payload["headers"]["x-tenant"], "one");
    assert_eq!(payload["credential_policy"], "strict");
}

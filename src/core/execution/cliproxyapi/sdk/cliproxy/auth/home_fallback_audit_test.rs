// ref: sdk/cliproxy/auth/home_fallback_audit_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: lifecycle-bound local fallback is disabled whenever Home owns dispatch
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use super::{Auth, AuthMutationOptions, HomeAuthRuntime};

#[test]
fn local_fallback_uses_manager_lifecycle_and_never_bypasses_live_home() {
    let manager = Arc::new(super::api_key_model_capabilities_test::manager());
    let mut auth = Auth::default();
    auth.id = "local-auth".into();
    auth.provider = "codex".into();
    manager
        .register(
            auth,
            AuthMutationOptions::default(),
            "2026-08-04T12:00:00Z".parse().unwrap(),
        )
        .unwrap();
    let runtime = HomeAuthRuntime::new(manager);
    assert_eq!(
        runtime.local_fallback_auth(" local-auth ").unwrap().id,
        "local-auth"
    );

    let transport = super::home_execution_paths_test::TestHomeTransport::with_auth_ids(&[]);
    let facade: Arc<dyn crate::internal::home::HomeTransport> = transport;
    let client = Arc::new(crate::internal::home::Client::new(
        crate::internal::home::HomeConfig {
            enabled: true,
            ..Default::default()
        },
        facade,
    ));
    runtime.publish_dispatch(
        client,
        Arc::new(crate::sdk::cliproxy::executionregistry::Registry::new()),
        1,
    );
    assert!(runtime.local_fallback_auth("local-auth").is_none());
}

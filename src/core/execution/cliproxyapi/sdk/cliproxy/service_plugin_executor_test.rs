// ref: sdk/cliproxy/service_plugin_executor_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::sync::Arc;

use super::service_executors::has_native_openai_compat_executor_config;
use super::service_test_support::{auth, registration, runtime_fixture, TestPluginRuntime};

#[test]
fn native_openai_compatibility_detection_uses_typed_auth_configuration() {
    let mut inline = auth("inline", "plugin-provider");
    inline
        .attributes
        .insert("base_url".into(), "https://compat.example/v1".into());
    assert!(has_native_openai_compat_executor_config(&inline));

    let mut metadata = auth("compat", "openai-compatibility");
    metadata
        .attributes
        .insert("compat_name".into(), "compat".into());
    assert!(has_native_openai_compat_executor_config(&metadata));

    let mut plugin = auth("plugin", "plugin-provider");
    plugin.attributes.insert("api_key".into(), "test".into());
    assert!(!has_native_openai_compat_executor_config(&plugin));
}

#[test]
fn plugin_candidate_displaces_only_native_fallback_registration() {
    let fixture = runtime_fixture(None);
    let native = registration("plugin-provider");
    fixture.runtime.auth_manager().register_executor(native);
    let plugin = Arc::new(TestPluginRuntime::default());
    plugin.add_candidate("plugin-provider");
    fixture.runtime.set_plugin_runtime(Some(plugin));

    fixture
        .runtime
        .ensure_executors_for_auth(&auth("plugin-auth", "plugin-provider"), false)
        .unwrap();
    assert!(fixture
        .runtime
        .auth_manager()
        .executors()
        .get("plugin-provider")
        .is_none());
}

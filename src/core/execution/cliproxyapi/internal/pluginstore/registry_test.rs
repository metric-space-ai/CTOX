// ref: internal/pluginstore/registry_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::pluginstore::{plugin_platforms, INSTALL_TYPE_DIRECT};

use super::registry::parse_registry;

#[test]
fn registry_parse_normalizes_and_validates_direct_plugins() {
    let registry = parse_registry(br#"{
      "schema_version": 2,
      "plugins": [{
        "id":" sample ", "name":" Sample ", "description":" Plugin ", "author":" Acme ", "version":"1.0.0",
        "install":{"type":"direct","artifacts":[{"goos":"macos","goarch":"aarch64","url":"https://example/sample.zip","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}]}
      }]
    }"#).unwrap();
    let plugin = registry.plugin_by_id("sample").unwrap();
    assert_eq!(plugin.name, "Sample");
    assert_eq!(plugin.install.install_type, INSTALL_TYPE_DIRECT);
    assert_eq!(plugin_platforms(plugin)[0].goos, "darwin");
    assert_eq!(plugin_platforms(plugin)[0].goarch, "arm64");
}

#[test]
fn registry_rejects_duplicate_and_schema_v1_direct() {
    let direct = r#"{"id":"sample","name":"Sample","description":"Plugin","author":"Acme","version":"1.0.0","install":{"type":"direct","artifacts":[{"goos":"linux","goarch":"amd64","url":"https://example/sample.zip","sha256":"0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}]}}"#;
    assert!(
        parse_registry(format!(r#"{{"schema_version":1,"plugins":[{direct}]}}"#).as_bytes())
            .unwrap_err()
            .to_string()
            .contains("schema_version 2")
    );
    assert!(parse_registry(
        format!(r#"{{"schema_version":2,"plugins":[{direct},{direct}]}}"#).as_bytes()
    )
    .unwrap_err()
    .to_string()
    .contains("duplicate"));
}

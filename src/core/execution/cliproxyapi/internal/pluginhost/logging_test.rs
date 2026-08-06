// ref: internal/pluginhost/logging_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use crate::sdk::pluginapi::Metadata;

use super::logging::{
    plugin_hot_reload_log_fields, plugin_log_fields, plugin_log_fields_from_metadata,
};

#[test]
fn plugin_log_fields_include_name_version_and_path() {
    let fields = plugin_log_fields_from_metadata(
        "sample",
        &Metadata {
            name: "Sample Provider".to_owned(),
            version: "0.2.0".to_owned(),
            ..Metadata::default()
        },
        "/tmp/plugins/sample-v0.2.0.dll",
    );

    assert_eq!(fields.get("plugin_id").map(String::as_str), Some("sample"));
    assert_eq!(
        fields.get("plugin_name").map(String::as_str),
        Some("Sample Provider")
    );
    assert_eq!(fields.get("version").map(String::as_str), Some("0.2.0"));
    assert_eq!(
        fields.get("path").map(String::as_str),
        Some("/tmp/plugins/sample-v0.2.0.dll")
    );
}

#[test]
fn plugin_log_fields_omit_empty_optional_values_and_trim_all_values() {
    let fields = plugin_log_fields(" sample ", " \t", " 0.2.0 ", "");
    assert_eq!(fields.get("plugin_id").map(String::as_str), Some("sample"));
    assert!(!fields.contains_key("plugin_name"));
    assert_eq!(fields.get("version").map(String::as_str), Some("0.2.0"));
    assert!(!fields.contains_key("path"));
}

#[test]
fn hot_reload_fields_include_active_and_retired_identity() {
    let fields = plugin_hot_reload_log_fields(
        "sample",
        "0.1.0",
        "/tmp/plugins/sample-v0.1.0.dll",
        "0.2.0",
        "/tmp/plugins/sample-v0.2.0.dll",
    );

    for (key, expected) in [
        ("plugin_id", "sample"),
        ("active_version", "0.1.0"),
        ("active_path", "/tmp/plugins/sample-v0.1.0.dll"),
        ("retired_version", "0.2.0"),
        ("retired_path", "/tmp/plugins/sample-v0.2.0.dll"),
    ] {
        assert_eq!(fields.get(key).map(String::as_str), Some(expected));
    }
}

#[test]
fn hot_reload_fields_omit_blank_optional_values_but_keep_plugin_id() {
    let fields = plugin_hot_reload_log_fields(" ", "", " \n", "", "");
    assert_eq!(fields.len(), 1);
    assert_eq!(fields.get("plugin_id").map(String::as_str), Some(""));
}

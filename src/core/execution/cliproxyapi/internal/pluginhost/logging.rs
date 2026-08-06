// ref: internal/pluginhost/logging.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::HashMap;

use crate::sdk::pluginapi::Metadata;

pub type PluginLogFields = HashMap<String, String>;

pub fn plugin_log_fields(id: &str, name: &str, version: &str, path: &str) -> PluginLogFields {
    let mut fields = HashMap::from([("plugin_id".to_owned(), id.trim().to_owned())]);
    for (key, value) in [("plugin_name", name), ("version", version), ("path", path)] {
        let value = value.trim();
        if !value.is_empty() {
            fields.insert(key.to_owned(), value.to_owned());
        }
    }
    fields
}

pub fn plugin_log_fields_from_metadata(
    id: &str,
    metadata: &Metadata,
    path: &str,
) -> PluginLogFields {
    plugin_log_fields(id, &metadata.name, &metadata.version, path)
}

pub fn plugin_hot_reload_log_fields(
    id: &str,
    active_version: &str,
    active_path: &str,
    retired_version: &str,
    retired_path: &str,
) -> PluginLogFields {
    let mut fields = HashMap::from([("plugin_id".to_owned(), id.trim().to_owned())]);
    for (key, value) in [
        ("active_version", active_version),
        ("active_path", active_path),
        ("retired_version", retired_version),
        ("retired_path", retired_path),
    ] {
        let value = value.trim();
        if !value.is_empty() {
            fields.insert(key.to_owned(), value.to_owned());
        }
    }
    fields
}

// ref: internal/pluginhost/support.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// Port-Note: support reflects LocalTransport availability, never cgo linkage
// License: MIT (upstream); modifications AGPL-3.0-only

pub fn support_plugin_header_value() -> &'static str {
    if cfg!(any(unix, windows)) {
        "1"
    } else {
        "0"
    }
}

pub const PLUGIN_MODE: &str = "process";

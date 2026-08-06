// ref: internal/pluginstore/manifest.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Manifest behavior is implemented on the public SDK types so the internal
//! I/O implementation and external callers validate one canonical contract.

pub use crate::sdk::pluginstore::{manifest_from_plugin, manifest_from_release, Manifest};

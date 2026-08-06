// ref: internal/pluginstore/home_sync.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Home synchronization DTOs and zeroization live at the SDK boundary; this
//! mirror retains the upstream package seam without duplicating secret state.

pub use crate::sdk::pluginstore::{PluginSyncItem, PluginSyncRequest, PluginSyncResponse};

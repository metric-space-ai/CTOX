// ref: internal/pluginhost/host_callbacks_unix.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: replaced_by_ctox
// Port-Note: cgo callback exports are replaced by versioned LocalTransport frames
// License: MIT (upstream); modifications AGPL-3.0-only

//! No symbols are exported into a plugin address space. Unix callbacks arrive
//! through [`super::process_transport`] and are authorized by
//! [`super::host_callbacks::HostCallbackRouter`].

pub const USES_IN_PROCESS_C_ABI: bool = false;

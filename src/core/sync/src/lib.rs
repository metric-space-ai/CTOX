//! Shared native execution authority for CTOX Sync.
//! Data replication and execution ownership are deliberately separate protocols.
pub mod authority;
pub mod checkpoint;
#[path = "contracts.generated.rs"]
pub mod contracts;
pub mod ipc;
#[cfg(unix)]
pub mod local_host;
#[cfg(feature = "webrtc")]
pub mod native;
#[cfg(feature = "webrtc")]
pub mod native_execution;

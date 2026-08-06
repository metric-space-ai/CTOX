// Origin: CTOX
// License: AGPL-3.0-only

mod auth;
mod checksum;
mod direct;
mod github;
mod home_sync;
mod install;
mod manifest;
mod registry;
mod version;

#[cfg(test)]
mod auth_test;
#[cfg(test)]
mod github_test;
#[cfg(test)]
mod home_sync_test;
#[cfg(test)]
mod install_test;
#[cfg(test)]
mod registry_test;
#[cfg(test)]
mod version_test;

pub use auth::{apply_resolved_auth, request_url_allowed, UrlPolicy};
pub use checksum::{parse_checksums, verify_checksum};
pub use github::{HttpRequest, HttpResponse, PluginStoreTransport, SafePluginStoreIo};
pub use home_sync::{PluginSyncItem, PluginSyncRequest, PluginSyncResponse};
pub use install::install_archive;
pub use manifest::{manifest_from_plugin, manifest_from_release, Manifest};
pub use version::update_available;

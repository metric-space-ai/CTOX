// ref: sdk/cliproxy/watcher.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

//! Watcher adapter corresponding to upstream `defaultWatcherFactory`.
//!
//! Upstream constructs the native watcher through a package global. CTOX
//! instead injects that constructor and retains the complete wrapper surface.

use std::path::Path;
use std::sync::Arc;

use super::types::{ReloadCallback, WatcherError, WatcherFactory, WatcherWrapper};

pub trait NativeWatcherFactory: Send + Sync {
    fn create_native(
        &self,
        config_path: &Path,
        auth_dir: &Path,
        reload: ReloadCallback,
    ) -> Result<WatcherWrapper, WatcherError>;
}

#[derive(Clone)]
pub struct DefaultWatcherFactory {
    native: Arc<dyn NativeWatcherFactory>,
}

#[must_use]
pub fn default_watcher_factory(native: Arc<dyn NativeWatcherFactory>) -> DefaultWatcherFactory {
    DefaultWatcherFactory { native }
}

impl WatcherFactory for DefaultWatcherFactory {
    fn create(
        &self,
        config_path: &Path,
        auth_dir: &Path,
        reload: ReloadCallback,
    ) -> Result<WatcherWrapper, WatcherError> {
        self.native.create_native(config_path, auth_dir, reload)
    }
}

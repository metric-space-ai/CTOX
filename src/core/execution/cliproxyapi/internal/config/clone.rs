// ref: internal/config/clone.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: ported
// License: MIT (upstream); modifications AGPL-3.0-only

use super::CliproxyRuntimeConfig;

impl CliproxyRuntimeConfig {
    /// Returns a fully independent runtime snapshot.
    ///
    /// Rust's owned configuration graph makes the reflection walker used by
    /// Go unnecessary: every reference-bearing field implements deep `Clone`.
    #[must_use]
    pub fn clone_for_runtime(&self) -> Self {
        self.clone()
    }
}

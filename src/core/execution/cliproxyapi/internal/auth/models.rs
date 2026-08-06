// ref: internal/auth/models.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::error::Error;
use std::path::Path;
use std::sync::{Arc, Mutex};

/// Provider token persistence boundary used during login and refresh flows.
///
/// CTOX keeps the interface path-oriented like upstream, but an implementation
/// must still be supplied by the typed secret/runtime owner. Merely possessing
/// an `Auth` record does not grant filesystem or secret-store authority.
pub trait TokenStorage: Send + Sync {
    fn save_token_to_file(&mut self, auth_file_path: &Path) -> Result<(), TokenStorageError>;
}

pub type TokenStorageError = Box<dyn Error + Send + Sync + 'static>;

/// Go interface values are shallow-copied by `Auth.Clone`. The Arc preserves
/// that shared implementation identity, while the mutex makes the upstream
/// mutable receiver explicit and serializes concurrent refresh writes.
pub type SharedTokenStorage = Arc<Mutex<Box<dyn TokenStorage>>>;

pub fn shared_token_storage(storage: impl TokenStorage + 'static) -> SharedTokenStorage {
    Arc::new(Mutex::new(Box::new(storage)))
}

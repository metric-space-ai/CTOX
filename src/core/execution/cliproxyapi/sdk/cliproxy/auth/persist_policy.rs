// ref: sdk/cliproxy/auth/persist_policy.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use super::{Auth, AuthSourceKind};

/// Typed replacement for upstream's private `context.Context` persistence key.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PersistenceIntent {
    #[default]
    Persist,
    /// The caller is projecting a record whose owning source already completed
    /// the durable write. This prevents file-watcher style write-back loops.
    SourceAlreadyPersisted,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct AuthMutationOptions {
    pub persistence: PersistenceIntent,
}

#[must_use]
pub fn should_persist(auth: &Auth, intent: PersistenceIntent) -> bool {
    if intent == PersistenceIntent::SourceAlreadyPersisted
        || auth.is_plugin_virtual()
        || matches!(
            auth.auth_source_kind(),
            Some(AuthSourceKind::Config | AuthSourceKind::Memory)
        )
    {
        return false;
    }

    // Go distinguishes nil metadata from an allocated empty map. Rust's owned
    // aggregate deliberately does not. An empty map carries no persisted
    // credential state, so CTOX treats it as runtime-only and avoids creating
    // an authority-free record.
    !auth.metadata.is_empty()
}

// Origin: CTOX
// License: AGPL-3.0-only

pub mod gitstore;
pub mod objectstore;
pub mod postgres_cooldown_store;
pub mod postgresstore;

pub use gitstore::{
    GitCredentialRef, GitPushRequest, GitRemoteRequest, GitStoreConfig, GitTokenStore,
    GitTransportAuthority, GitTransportError,
};
pub use objectstore::{ObjectBackend, ObjectEntry, ObjectStoreConfig, ObjectTokenStore};
pub use postgres_cooldown_store::PostgresCooldownStateStore;
pub use postgresstore::{
    CooldownMutation, CooldownRow, PostgresBackend, PostgresStore, PostgresStoreConfig, StoreError,
    StoredAuthRecord,
};

#[cfg(test)]
mod gitstore_test;
#[cfg(test)]
mod postgres_cooldown_store_test;

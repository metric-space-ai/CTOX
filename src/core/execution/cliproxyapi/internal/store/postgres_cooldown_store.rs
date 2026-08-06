// ref: internal/store/postgres_cooldown_store.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::sdk::cliproxy::auth::{CooldownStateRecord, CooldownStateStore, CooldownStoreError};

use super::postgresstore::{
    CooldownMutation, CooldownRow, PostgresBackend, PostgresStore, StoreError,
};

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct CooldownKey {
    auth_id: String,
    model: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct CooldownVersion {
    updated_at_ms: i64,
}

/// Instance-local observation cursor over a shared, atomic backend. The cursor
/// intentionally mirrors upstream: a save only tombstones records observed by
/// this instance, and the backend's compare-and-set guard protects concurrent
/// newer rows.
pub struct PostgresCooldownStateStore {
    backend: std::sync::Arc<dyn PostgresBackend>,
    previous: Mutex<BTreeMap<CooldownKey, CooldownVersion>>,
}

impl std::fmt::Debug for PostgresCooldownStateStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("PostgresCooldownStateStore")
            .finish_non_exhaustive()
    }
}

impl PostgresCooldownStateStore {
    #[must_use]
    pub fn new(store: &PostgresStore) -> Self {
        Self {
            backend: store.backend().clone(),
            previous: Mutex::new(BTreeMap::new()),
        }
    }

    pub fn load_records(&self) -> Result<Vec<CooldownStateRecord>, StoreError> {
        let mut records = Vec::new();
        let mut previous = BTreeMap::new();
        for row in self.backend.load_cooldowns()? {
            if row.deleted {
                continue;
            }
            let record: CooldownStateRecord = serde_json::from_slice(&row.content)
                .map_err(|error| StoreError::Serialization(error.to_string()))?;
            record
                .validate()
                .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
            let key = cooldown_key(&record)?;
            previous.insert(
                key,
                CooldownVersion {
                    updated_at_ms: row.updated_at_ms,
                },
            );
            records.push(record);
        }
        records.sort_by(|left, right| {
            cooldown_key(left)
                .expect("validated cooldown")
                .cmp(&cooldown_key(right).expect("validated cooldown"))
        });
        *self
            .previous
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = previous;
        Ok(records)
    }

    pub fn save_records(&self, records: &[CooldownStateRecord]) -> Result<(), StoreError> {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|duration| i64::try_from(duration.as_millis()).unwrap_or(i64::MAX))
            .unwrap_or_default();
        let mut current = BTreeMap::new();
        let mut mutations = Vec::with_capacity(records.len());
        for input in records {
            input
                .validate()
                .map_err(|error| StoreError::InvalidRecord(error.to_string()))?;
            let mut record = input.clone();
            if record.updated_at_ms == 0 {
                record.updated_at_ms = now_ms;
            }
            let key = cooldown_key(&record)?;
            if current.contains_key(&key) {
                return Err(StoreError::InvalidRecord(format!(
                    "duplicate cooldown identity {}/{}",
                    key.auth_id, key.model
                )));
            }
            let content = serde_json::to_vec(&record)
                .map_err(|error| StoreError::Serialization(error.to_string()))?;
            current.insert(
                key.clone(),
                CooldownVersion {
                    updated_at_ms: record.updated_at_ms,
                },
            );
            mutations.push(CooldownMutation {
                row: CooldownRow {
                    auth_id: key.auth_id,
                    model: key.model,
                    content,
                    deleted: false,
                    updated_at_ms: record.updated_at_ms,
                },
                delete_if_not_newer_than_ms: None,
            });
        }

        let mut previous_guard = self
            .previous
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for (key, previous) in previous_guard.iter() {
            if current.contains_key(key) {
                continue;
            }
            let deleted_at_ms = now_ms.max(previous.updated_at_ms.saturating_add(1));
            mutations.push(CooldownMutation {
                row: CooldownRow {
                    auth_id: key.auth_id.clone(),
                    model: key.model.clone(),
                    content: b"{}".to_vec(),
                    deleted: true,
                    updated_at_ms: deleted_at_ms,
                },
                delete_if_not_newer_than_ms: Some(previous.updated_at_ms),
            });
        }
        self.backend.apply_cooldown_mutations(&mutations)?;
        *previous_guard = current;
        Ok(())
    }
}

impl CooldownStateStore for PostgresCooldownStateStore {
    fn load(&self) -> Result<Vec<CooldownStateRecord>, CooldownStoreError> {
        self.load_records().map_err(|_| CooldownStoreError::Read)
    }

    fn save(&self, records: &[CooldownStateRecord]) -> Result<(), CooldownStoreError> {
        self.save_records(records).map_err(|error| match error {
            StoreError::InvalidRecord(_) => CooldownStoreError::InvalidRecord,
            _ => CooldownStoreError::Write,
        })
    }
}

fn cooldown_key(record: &CooldownStateRecord) -> Result<CooldownKey, StoreError> {
    let auth_id = record.auth_id.trim();
    let model = record.model.as_deref().unwrap_or_default().trim();
    if auth_id.is_empty() {
        return Err(StoreError::InvalidRecord(
            "cooldown state has empty auth ID".to_owned(),
        ));
    }
    Ok(CooldownKey {
        auth_id: auth_id.to_owned(),
        model: model.to_owned(),
    })
}

impl PostgresStore {
    #[must_use]
    pub fn cooldown_state_store(&self) -> PostgresCooldownStateStore {
        PostgresCooldownStateStore::new(self)
    }
}

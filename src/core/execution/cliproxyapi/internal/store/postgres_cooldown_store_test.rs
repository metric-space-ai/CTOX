// ref: internal/store/postgres_cooldown_store_test.go @ a88197f845c979132c8978ea223c6af05cc81536
// Port-Status: adapted_to_ctox
// License: MIT (upstream); modifications AGPL-3.0-only

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use serde_json::{json, Value};
use tempfile::TempDir;

use crate::sdk::cliproxy::auth::{Auth, CooldownStateRecord};

use super::objectstore::{ObjectBackend, ObjectEntry, ObjectStoreConfig, ObjectTokenStore};
use super::postgres_cooldown_store::PostgresCooldownStateStore;
use super::postgresstore::{
    CooldownMutation, CooldownRow, PostgresBackend, PostgresStore, PostgresStoreConfig, StoreError,
    StoredAuthRecord,
};

#[derive(Default)]
struct MemoryPostgres {
    config: Mutex<BTreeMap<String, Vec<u8>>>,
    auth: Mutex<BTreeMap<String, StoredAuthRecord>>,
    cooldown: Mutex<BTreeMap<(String, String), CooldownRow>>,
}

impl PostgresBackend for MemoryPostgres {
    fn ensure_schema(&self, _: &PostgresStoreConfig) -> Result<(), StoreError> {
        Ok(())
    }

    fn load_config(&self, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self.config.lock().unwrap().get(key).cloned())
    }

    fn put_config(&self, key: &str, content: &[u8]) -> Result<(), StoreError> {
        self.config
            .lock()
            .unwrap()
            .insert(key.to_owned(), content.to_vec());
        Ok(())
    }

    fn delete_config(&self, key: &str) -> Result<(), StoreError> {
        self.config.lock().unwrap().remove(key);
        Ok(())
    }

    fn list_auth(&self) -> Result<Vec<StoredAuthRecord>, StoreError> {
        Ok(self.auth.lock().unwrap().values().cloned().collect())
    }

    fn put_auth(&self, id: &str, content: &[u8]) -> Result<(), StoreError> {
        let now = DateTime::<Utc>::from(std::time::UNIX_EPOCH);
        self.auth.lock().unwrap().insert(
            id.to_owned(),
            StoredAuthRecord {
                id: id.to_owned(),
                content: content.to_vec(),
                created_at: now,
                updated_at: now,
            },
        );
        Ok(())
    }

    fn delete_auth(&self, id: &str) -> Result<(), StoreError> {
        self.auth.lock().unwrap().remove(id);
        Ok(())
    }

    fn load_cooldowns(&self) -> Result<Vec<CooldownRow>, StoreError> {
        Ok(self.cooldown.lock().unwrap().values().cloned().collect())
    }

    fn apply_cooldown_mutations(&self, mutations: &[CooldownMutation]) -> Result<(), StoreError> {
        let mut rows = self.cooldown.lock().unwrap();
        for mutation in mutations {
            let key = (mutation.row.auth_id.clone(), mutation.row.model.clone());
            match mutation.delete_if_not_newer_than_ms {
                Some(observed) => {
                    let may_delete = rows
                        .get(&key)
                        .is_some_and(|row| !row.deleted && row.updated_at_ms <= observed);
                    if may_delete {
                        rows.insert(key, mutation.row.clone());
                    }
                }
                None => {
                    let may_write = rows
                        .get(&key)
                        .is_none_or(|row| row.updated_at_ms <= mutation.row.updated_at_ms);
                    if may_write {
                        rows.insert(key, mutation.row.clone());
                    }
                }
            }
        }
        Ok(())
    }
}

fn postgres_store(root: &Path, backend: Arc<MemoryPostgres>) -> PostgresStore {
    PostgresStore::new(
        PostgresStoreConfig {
            schema: "cliproxy".to_owned(),
            config_table: String::new(),
            auth_table: String::new(),
            cooldown_table: String::new(),
            spool_dir: root.to_path_buf(),
        },
        backend,
    )
    .unwrap()
}

fn cooldown(auth_id: &str, model: Option<&str>, updated_at_ms: i64) -> CooldownStateRecord {
    CooldownStateRecord {
        provider: "codex".to_owned(),
        auth_id: auth_id.to_owned(),
        model: model.map(str::to_owned),
        status: "cooling".to_owned(),
        next_retry_after_ms: Some(updated_at_ms + 1_000),
        reason: "rate_limit".to_owned(),
        quota: Default::default(),
        last_error: None,
        updated_at_ms,
    }
}

#[test]
fn postgres_cooldown_save_load_and_tombstone_round_trip() {
    let root = TempDir::new().unwrap();
    let backend = Arc::new(MemoryPostgres::default());
    let store = postgres_store(root.path(), backend.clone());
    let cooldowns = PostgresCooldownStateStore::new(&store);
    let first = cooldown("account-a", Some("model-a"), 1_000);
    let second = cooldown("account-b", None, 2_000);

    cooldowns
        .save_records(&[first.clone(), second.clone()])
        .unwrap();
    assert_eq!(
        cooldowns.load_records().unwrap(),
        vec![first.clone(), second]
    );
    cooldowns
        .save_records(std::slice::from_ref(&first))
        .unwrap();
    assert_eq!(cooldowns.load_records().unwrap(), vec![first]);
    assert!(
        backend
            .cooldown
            .lock()
            .unwrap()
            .get(&("account-b".to_owned(), String::new()))
            .unwrap()
            .deleted
    );
}

#[test]
fn postgres_cooldown_concurrent_instances_merge_and_reject_stale_delete() {
    let root = TempDir::new().unwrap();
    let backend = Arc::new(MemoryPostgres::default());
    let store = postgres_store(root.path(), backend);
    let store_a = PostgresCooldownStateStore::new(&store);
    let store_b = PostgresCooldownStateStore::new(&store);
    let stale = PostgresCooldownStateStore::new(&store);
    assert!(store_a.load_records().unwrap().is_empty());
    assert!(store_b.load_records().unwrap().is_empty());

    let a = cooldown("account-a", Some("model-a"), 1_000);
    let b = cooldown("account-b", Some("model-b"), 1_000);
    store_a.save_records(std::slice::from_ref(&a)).unwrap();
    store_b.save_records(std::slice::from_ref(&b)).unwrap();
    let observed = stale.load_records().unwrap();
    assert_eq!(observed.len(), 2);

    let mut newer_a = a;
    newer_a.updated_at_ms = 5_000;
    store_a
        .save_records(std::slice::from_ref(&newer_a))
        .unwrap();
    stale.save_records(std::slice::from_ref(&b)).unwrap();
    let loaded = PostgresCooldownStateStore::new(&store)
        .load_records()
        .unwrap();
    assert_eq!(loaded.len(), 2, "stale tombstone must not erase newer A");

    store_a.save_records(&[]).unwrap();
    stale.save_records(&observed).unwrap();
    let loaded = PostgresCooldownStateStore::new(&store)
        .load_records()
        .unwrap();
    assert_eq!(loaded.len(), 1);
    assert_eq!(loaded[0].auth_id, "account-b");
}

#[test]
fn postgres_spool_bootstrap_save_list_delete_and_config_are_atomic_and_bounded() {
    let root = TempDir::new().unwrap();
    let backend = Arc::new(MemoryPostgres::default());
    backend
        .put_config("config", b"first\r\nsecond\rthird\n")
        .unwrap();
    backend
        .put_auth(
            "team/one.json",
            br#"{"type":"codex","email":"a@example.test"}"#,
        )
        .unwrap();
    let store = postgres_store(root.path(), backend.clone());
    store.bootstrap(None).unwrap();
    assert_eq!(
        fs::read(store.config_path()).unwrap(),
        b"first\nsecond\nthird\n"
    );
    assert!(store.auth_dir().join("team/one.json").exists());

    let mut auth = Auth::default();
    auth.id = "team/two.json".to_owned();
    auth.metadata = BTreeMap::from([
        ("type".to_owned(), Value::String("claude".to_owned())),
        ("label".to_owned(), Value::String("Team Two".to_owned())),
    ]);
    let path = store.save(&mut auth).unwrap();
    assert_eq!(path, store.auth_dir().join("team/two.json"));
    assert_eq!(auth.attributes.get("source_backend").unwrap(), "postgres");
    let listed = store.list().unwrap();
    assert_eq!(listed.len(), 2);
    assert!(listed.iter().any(|entry| entry.label == "Team Two"));

    fs::write(store.config_path(), b"new\r\nvalue").unwrap();
    store.persist_config().unwrap();
    assert_eq!(
        backend.load_config("config").unwrap().unwrap(),
        b"new\nvalue"
    );
    store.delete("team/two.json").unwrap();
    assert!(!path.exists());
    assert_eq!(store.list().unwrap().len(), 1);
    assert!(store.delete("../escape.json").is_err());
}

#[derive(Default)]
struct MemoryObjects(Mutex<BTreeMap<String, Vec<u8>>>);

impl ObjectBackend for MemoryObjects {
    fn ensure_bucket(&self, _: &str, _: &str) -> Result<(), StoreError> {
        Ok(())
    }

    fn get(&self, _: &str, key: &str) -> Result<Option<Vec<u8>>, StoreError> {
        Ok(self.0.lock().unwrap().get(key).cloned())
    }

    fn list(&self, _: &str, prefix: &str) -> Result<Vec<ObjectEntry>, StoreError> {
        Ok(self
            .0
            .lock()
            .unwrap()
            .iter()
            .filter(|(key, _)| key.starts_with(prefix))
            .map(|(key, value)| ObjectEntry {
                key: key.clone(),
                content: value.clone(),
            })
            .collect())
    }

    fn put(&self, _: &str, key: &str, content: &[u8], _: &str) -> Result<(), StoreError> {
        self.0
            .lock()
            .unwrap()
            .insert(key.to_owned(), content.to_vec());
        Ok(())
    }

    fn delete(&self, _: &str, key: &str) -> Result<(), StoreError> {
        self.0.lock().unwrap().remove(key);
        Ok(())
    }
}

#[test]
fn object_store_bootstrap_is_incremental_and_round_trips_auth_and_config() {
    let root = TempDir::new().unwrap();
    let backend = Arc::new(MemoryObjects::default());
    backend
        .0
        .lock()
        .unwrap()
        .insert("tenant/config/config.yaml".to_owned(), b"a\r\nb\r".to_vec());
    backend.0.lock().unwrap().insert(
        "tenant/auths/team/one.json".to_owned(),
        serde_json::to_vec(&json!({"type":"codex"})).unwrap(),
    );
    let store = ObjectTokenStore::new(
        ObjectStoreConfig {
            bucket: "bucket".to_owned(),
            region: "eu-test-1".to_owned(),
            prefix: "/tenant/".to_owned(),
            local_root: root.path().join("objects"),
        },
        backend.clone(),
    )
    .unwrap();
    fs::write(
        store.auth_dir().join("local-only.json"),
        br#"{"type":"claude"}"#,
    )
    .unwrap();
    store.bootstrap(None).unwrap();
    assert!(store.auth_dir().join("local-only.json").exists());
    assert!(store.auth_dir().join("team/one.json").exists());
    assert_eq!(fs::read(store.config_path()).unwrap(), b"a\nb\n");

    let mut auth = Auth::default();
    auth.id = "nested/two".to_owned();
    auth.metadata.insert("type".to_owned(), json!("claude"));
    let saved = store.save(&mut auth).unwrap();
    assert_eq!(saved, store.auth_dir().join("nested/two.json"));
    assert!(backend
        .0
        .lock()
        .unwrap()
        .contains_key("tenant/auths/nested/two.json"));
    store.delete("nested/two").unwrap();
    assert!(!backend
        .0
        .lock()
        .unwrap()
        .contains_key("tenant/auths/nested/two.json"));
    assert!(store.delete("../../escape").is_err());
}

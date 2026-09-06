use ctox_sync::native::{NativeAdmission, NativeSyncOptions};
use rxdb::{
    plugins::replication_webrtc::webrtc_types::WebRTCPeerSessionValidation,
    rx_database::{create_rx_database, RxCollectionCreator, RxDatabase, RxDatabaseCreator},
    storage::sqlite::{index_mod::get_rx_storage_sqlite, types::RxStorageSqliteSettings},
    types::{HashFunction, HashOutput},
};
use serde_json::json;
use std::{collections::HashMap, sync::Arc, time::Duration};
struct Hash;
impl HashFunction for Hash {
    fn hash<'a>(&'a self, input: String) -> HashOutput<'a> {
        Box::pin(async move { rxdb::plugins::utils::utils_hash::native_sha256(&input) })
    }
}
pub async fn options(
    url: String,
    room: &str,
    session: &str,
) -> (tempfile::TempDir, Arc<RxDatabase>, NativeSyncOptions) {
    let directory = tempfile::tempdir().unwrap();
    let database = create_rx_database(RxDatabaseCreator {
        name: format!(
            "native-{}",
            directory.path().file_name().unwrap().to_string_lossy()
        ),
        storage: get_rx_storage_sqlite(RxStorageSqliteSettings {
            database_path: directory.path().join("records.sqlite"),
        }),
        multi_instance: false,
        password: None,
        hash_function: Arc::new(Hash),
        options: HashMap::new(),
        ignore_duplicate: false,
        close_duplicates: false,
        event_reduce: false,
        allow_slow_count: false,
    })
    .await
    .unwrap();
    let collections = database
        .add_collections(HashMap::from([(
            "records".into(),
            RxCollectionCreator {
                schema: serde_json::from_value(
                    json!({"version":0,"primaryKey":"id","type":"object",
            "properties":{"id":{"type":"string","maxLength":64}},"required":["id"]}),
                )
                .unwrap(),
                conflict_handler: None,
                options: HashMap::new(),
            },
        )]))
        .await
        .unwrap();
    let options = NativeSyncOptions {
        peer_role: ctox_sync::native::NativePeerRole::CtoxInstance,
        collections: collections.into_values().collect(),
        signaling_urls: Arc::new(move || vec![url.clone()]),
        room: room.into(),
        peer_session_id: session.into(),
        // No external STUN/TURN endpoints in an isolated localhost fixture.
        ice_servers: vec![Default::default()],
        admission: NativeAdmission {
            peer: Arc::new(|_| true),
            session: Arc::new(|payload, _| {
                if payload
                    .pointer("/peerSession/role")
                    .and_then(serde_json::Value::as_str)
                    == Some("ctox_instance")
                {
                    WebRTCPeerSessionValidation::Accept
                } else {
                    WebRTCPeerSessionValidation::Reject
                }
            }),
            collection_read: Some(Arc::new(|_, _| false)),
            collection_write: Some(Arc::new(|_, _| false)),
            document_read: None,
            document_write: None,
            eager_pull: None,
            live_change: None,
        },
        bringup_timeout: Duration::from_secs(5),
    };
    (directory, database, options)
}

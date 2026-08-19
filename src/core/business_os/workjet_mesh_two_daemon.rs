// Origin: CTOX
// License: AGPL-3.0-only

//! The decisive two-daemon proof for the Workjet mailbox mesh.
//!
//! Everything below the `#[cfg(test)]` marker is test-only; this module carries
//! no production code. It exists as its own module because the harness it needs
//! — a real signaling server, two storage roots, two RxDB databases and two live
//! WebRTC replication sessions — has nothing to do with the units in
//! `workjet_mesh`/`workjet_mesh_join`, whose tests are deliberately fast and
//! hermetic.
//!
//! # What is actually proven
//!
//! The unit tests prove that an invite parses, that a membership persists
//! owner-only, that status redacts, and that the collection selection is scoped
//! to the mailbox. None of that proves the thing the feature exists for: that a
//! CTOX daemon can JOIN A FOREIGN DAEMON'S ROOM and exchange envelopes with it.
//! Specifically, four claims can only be checked end to end:
//!
//! 1. The joining peer satisfies the SERVING side's signaling-room password.
//!    The signaling server partitions rooms by `token|roomId`, so a joiner that
//!    derives the token from the wrong password lands in a different partition
//!    and simply never sees the server — a silent no-op, not an error.
//! 2. The joining peer satisfies the serving daemon's per-collection READ authz,
//!    which is evaluated against `peerSession.capabilityToken`. This is the
//!    primitive commit b1164213 added; without it the session connects, hands
//!    shakes, and replicates nothing — the failure mode that looks healthy.
//! 3. Replication is bidirectional. A push-only mesh passes any one-way test.
//! 4. Deletes replicate as tombstones, not as "the row is gone at A and stays
//!    at B forever".
//!
//! # Why writes go through `publish_envelope`
//!
//! `publish_envelope` writes the row with plain SQL, exactly as the Workjet
//! loopback intake does. Writing through the `RxCollection` handle instead would
//! test replication but NOT the loopback surface's interaction with it, and the
//! loopback surface is the only way an envelope ever enters this table in
//! production. The rxdb SQLite storage detects the external write through its
//! `__rxdb_changed_tables` trigger counter; if that path did not work, the mesh
//! would be decorative, so the test depends on it on purpose.

#[cfg(test)]
mod tests {
    use std::net::TcpStream;
    use std::path::Path;
    use std::path::PathBuf;
    use std::process::Child;
    use std::process::Command;
    use std::process::Stdio;
    use std::sync::Arc;
    use std::time::Duration;
    use std::time::Instant;

    use rxdb::plugins::replication_webrtc::CollectionAuthzHook;
    use rxdb::plugins::replication_webrtc::DocumentReadAuthzHook;
    use rxdb::plugins::replication_webrtc::DocumentWriteAuthzHook;
    use rxdb::rx_collection::RxCollection;
    use serde_json::json;
    use serde_json::Value;

    use crate::business_os::rxdb_peer;
    use crate::business_os::store;
    use crate::business_os::threads;
    use crate::business_os::workjet_mailbox;
    use crate::business_os::workjet_mesh;
    use crate::business_os::workjet_mesh_cli;
    use crate::business_os::workjet_mesh_join;

    /// Whole-test budget. The brief caps the run at 60s; the assertions below
    /// each carry a smaller deadline so a failure reports WHICH leg stalled
    /// instead of dying at the harness timeout.
    const CONNECT_DEADLINE: Duration = Duration::from_secs(30);
    const REPLICATION_DEADLINE: Duration = Duration::from_secs(20);
    const POLL_INTERVAL: Duration = Duration::from_millis(250);

    /// Kills the signaling server on drop, including on assertion panic — a
    /// leaked node process would hold the port and wedge the next run.
    struct SignalingServer {
        child: Child,
        port: u16,
        /// `SIGNALING_DEBUG=1` transcript. When a leg of the test stalls, this
        /// file is the only place that says WHETHER the two peers ever met in
        /// the same room partition — a token or instance mismatch looks exactly
        /// like a healthy-but-silent session from inside the process.
        log: PathBuf,
    }

    impl SignalingServer {
        fn url(&self) -> String {
            format!("ws://127.0.0.1:{}", self.port)
        }

        fn transcript(&self) -> String {
            std::fs::read_to_string(&self.log).unwrap_or_else(|err| format!("<unreadable: {err}>"))
        }
    }

    impl Drop for SignalingServer {
        fn drop(&mut self) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    }

    fn node_available() -> bool {
        Command::new("node")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|status| status.success())
            .unwrap_or(false)
    }

    fn free_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").expect("bind ephemeral port");
        let port = listener.local_addr().expect("local addr").port();
        drop(listener);
        port
    }

    fn start_signaling_server() -> Option<SignalingServer> {
        let script = repo_root().join("src/core/rxdb/tools/local_signaling_server.js");
        if !script.exists() {
            return None;
        }
        let port = free_port();
        let log = std::env::temp_dir().join(format!("ctox-mesh-signaling-{port}.log"));
        let sink = std::fs::File::create(&log).ok()?;
        let child = Command::new("node")
            .arg(&script)
            .env("SIGNALING_HOST", "127.0.0.1")
            .env("SIGNALING_PORT", port.to_string())
            .env("SIGNALING_DEBUG", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::from(sink))
            .spawn()
            .ok()?;
        let server = SignalingServer { child, port, log };
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            if TcpStream::connect(("127.0.0.1", port)).is_ok() {
                return Some(server);
            }
            std::thread::sleep(POLL_INTERVAL);
        }
        None
    }

    /// Points a storage root's sync configuration at the local signaling server.
    /// Written as the persisted runtime file rather than through
    /// `CTOX_BUSINESS_OS_SIGNALING_URLS`, because the env var is process-global
    /// and this test runs TWO roots that must stay independently configurable.
    fn pin_signaling_url(root: &Path, url: &str) {
        let dir = root.join("runtime");
        std::fs::create_dir_all(&dir).expect("runtime dir");
        std::fs::write(
            dir.join("business-os-signaling-urls.json"),
            serde_json::to_vec_pretty(&json!([url])).expect("render"),
        )
        .expect("pin signaling url");
        // Both peers are on loopback, so host candidates are sufficient and a
        // reflexive candidate is meaningless. The default configuration points
        // at a PUBLIC STUN server; on an offline or sandboxed machine ICE
        // gathering then blocks on that server's timeout and the whole session
        // stalls before an offer is ever sent. Pinning ICE to a closed local
        // port makes gathering fail fast and deterministically instead. This
        // relaxes nothing about authorization — it only removes a network
        // dependency the mesh does not need on loopback.
        crate::inference::runtime_env::set_runtime_env_value(
            root,
            "CTOX_BUSINESS_OS_ICE_SERVERS",
            r#"[{"urls":"stun:127.0.0.1:3478"}]"#,
        )
        .expect("pin ice servers");
    }

    /// Opens a root's Business OS database with ONLY the mailbox collection.
    ///
    /// The daemon registers ~90 collections here; registering them all would put
    /// the schema work, migration sweep and index build of two full instances
    /// inside a 60s budget for no added evidence. What the mesh needs from the
    /// database is exactly one live `RxCollection` over the real mailbox schema
    /// and the real SQLite table, which is what this produces.
    async fn open_mailbox_collection(root: &Path) -> Arc<RxCollection> {
        let database = rxdb_peer::open_database(store::rxdb_store_path(root))
            .await
            .expect("open database");
        let creators = workjet_mailbox::with_mailbox_collection(Default::default());
        let (collections, failed) = database
            .add_collections_tolerant(creators)
            .await
            .expect("register mailbox collection");
        assert!(failed.is_empty(), "mailbox registration failed: {failed:?}");
        store::ensure_legacy_collection_grants(root, &[workjet_mesh::MESH_COLLECTION.to_string()])
            .expect("collection grants");
        // Leak the database handle for the duration of the test: dropping it
        // would close the storage under the live replication pool.
        let database = Box::leak(Box::new(database));
        let _ = &*database;
        collections
            .into_iter()
            .find(|(name, _)| name == workjet_mesh::MESH_COLLECTION)
            .expect("mailbox collection registered")
            .1
    }

    /// Brings up the SERVING side exactly as `rxdb_peer::run_native_peer` does:
    /// same URL provider, same revocation validators, and — decisively — the
    /// same per-collection and per-document authz hooks. Weakening any of them
    /// here would make the test prove nothing about whether a real serving
    /// daemon accepts the join.
    async fn serve_room(root: &Path, collection: Arc<RxCollection>) -> rxdb_peer::WebRtcPool {
        let sync = store::sync_config(root).expect("sync config");
        let peer_session_id = format!("rxdb-rs-serve-{}", uuid::Uuid::new_v4().simple());
        let url_provider = rxdb_peer::native_signaling_url_provider(
            sync.signaling_urls.clone(),
            sync.sync_room.clone(),
            sync.signaling_room_password.clone(),
            peer_session_id.clone(),
        );
        let ice_servers = rxdb_peer::ice_servers_from_sync_config(&sync.ice_servers);

        let authz_root = root.to_path_buf();
        let collection_authz: Option<CollectionAuthzHook> =
            Some(Arc::new(move |token: &str, collection: &str| {
                store::capability_allows_collection_permission(
                    &authz_root,
                    token,
                    collection,
                    crate::business_os::policy::BusinessOsPermission::DataRead,
                )
            }));
        let write_root = root.to_path_buf();
        let collection_write_authz: Option<CollectionAuthzHook> =
            Some(Arc::new(move |token: &str, collection: &str| {
                threads::may_accept_peer_write(&write_root, token, collection)
            }));
        let read_filter_root = root.to_path_buf();
        let document_read_authz: Option<DocumentReadAuthzHook> =
            Some(Arc::new(move |token: &str, collection: &str| {
                threads::replication_document_filter(&read_filter_root, token, collection)
            }));
        let doc_write_root = root.to_path_buf();
        let document_write_authz: Option<DocumentWriteAuthzHook> = Some(Arc::new(
            move |token: &str, collection: &str, document: &Value| {
                threads::may_accept_peer_document_write(
                    &doc_write_root,
                    token,
                    collection,
                    document,
                )
            },
        ));

        rxdb::plugins::replication_webrtc::replicate_web_rtc_rs_multi_with_url_list_provider_and_validators(
            vec![collection],
            url_provider,
            sync.sync_room.clone(),
            peer_session_id,
            ice_servers,
            None,
            None,
            collection_authz,
            collection_write_authz,
            document_read_authz,
            document_write_authz,
            20,
            20,
            5_000,
        )
        .await
        .expect("serving replication pool")
    }

    /// Builds the exact document `ctox business-os desktop invite` emits, with a
    /// REAL store-issued capability token — a synthetic token would be rejected
    /// by the serving side's read authz and the test would prove the opposite of
    /// what it claims.
    fn generate_invite(root: &Path) -> Value {
        let sync = store::sync_config(root).expect("sync config");
        let (capability_token, capability_expires_at_ms) =
            store::issue_business_os_capability_token_for_managed_user(
                root,
                "mesh-peer",
                "Mesh Peer",
                "chef",
                workjet_mailbox::now_ms(),
            )
            .expect("issue capability token");
        json!({
            "type": "ctox-business-os-invite",
            "version": 1,
            "display_name": "Machine A",
            "instance_id": sync.instance_id,
            "sync_room": sync.sync_room,
            "signaling_urls": sync.signaling_urls,
            "signaling_room_password": sync.signaling_room_password,
            "transport": "webrtc",
            "data_plane": "rxdb-webrtc",
            "expires_at": chrono::DateTime::from_timestamp_millis(capability_expires_at_ms)
                .expect("capability expiry")
                .to_rfc3339_opts(chrono::SecondsFormat::Millis, true),
            "session": {
                "authenticated": true,
                "source": "desktop_invite",
                "capability_token": capability_token,
                "user": { "id": "mesh-peer", "role": "chef" }
            }
        })
    }

    fn publish(root: &Path, id: &str, target: &str, expires_at_ms: i64) {
        let result = workjet_mailbox::publish_envelope(
            root,
            &json!({
                "id": id,
                "target_environment_id": target,
                "expires_at_ms": expires_at_ms,
                "envelope_json": json!({ "id": id, "kind": "mesh-probe" }).to_string(),
            }),
        )
        .expect("publish envelope");
        assert_eq!(result["ok"], Value::Bool(true), "{result}");
    }

    /// Row-level dump of a root's mailbox table: `id deleted rev lwt`.
    ///
    /// A replication leg can fail in two very different ways that look
    /// identical from the id lists alone — the row never arrived, or it arrived
    /// and lost conflict resolution. The revision height and the write time are
    /// what distinguish them.
    fn mailbox_rows(root: &Path) -> Vec<String> {
        let Ok(conn) = workjet_mailbox::open_mailbox_store(root) else {
            return vec!["<mailbox store unreadable>".to_string()];
        };
        let table = workjet_mailbox::MAILBOX_TABLE;
        let Ok(mut statement) = conn.prepare(&format!(
            r#"SELECT id, deleted, revision, lastWriteTime FROM "{table}" ORDER BY id"#
        )) else {
            return vec!["<mailbox query failed>".to_string()];
        };
        statement
            .query_map([], |row| {
                Ok(format!(
                    "{} deleted={} rev={} lwt={}",
                    row.get::<_, String>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                    row.get::<_, f64>(3)?,
                ))
            })
            .and_then(|rows| rows.collect::<Result<Vec<_>, _>>())
            .unwrap_or_else(|err| vec![format!("<row read failed: {err}>")])
    }

    /// `(live, tombstoned)` ids in a root's mailbox table — the exact view a
    /// Workjet transport gets from the loopback pending route.
    fn envelope_ids(root: &Path) -> (Vec<String>, Vec<String>) {
        let conn = workjet_mailbox::open_mailbox_store(root).expect("open mailbox store");
        let table = workjet_mailbox::MAILBOX_TABLE;
        let mut read = |deleted: i64| -> Vec<String> {
            let mut statement = conn
                .prepare(&format!(
                    r#"SELECT id FROM "{table}" WHERE deleted = ?1 ORDER BY id"#
                ))
                .expect("prepare");
            let rows = statement
                .query_map([deleted], |row| row.get::<_, String>(0))
                .expect("query")
                .collect::<Result<Vec<_>, _>>()
                .expect("rows");
            rows
        };
        (read(0), read(1))
    }

    async fn await_condition(
        deadline: Duration,
        what: &str,
        mut ready: impl FnMut() -> bool,
    ) -> bool {
        let stop = Instant::now() + deadline;
        while Instant::now() < stop {
            if ready() {
                return true;
            }
            tokio::time::sleep(POLL_INTERVAL).await;
        }
        eprintln!("[mesh-test] timed out waiting for {what}");
        false
    }

    /// THE decisive test: two storage roots, two rooms, one mesh edge.
    ///
    /// Skips (rather than fails) when `node` is unavailable, because the
    /// signaling server is a node script and a machine without node cannot run
    /// ANY of the repository's WebRTC evidence. The skip is loud on stderr.
    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn two_daemons_replicate_only_the_mailbox_across_a_mesh_join() {
        if !node_available() {
            eprintln!("[mesh-test] SKIPPED: `node` is not on PATH");
            return;
        }
        let Some(signaling) = start_signaling_server() else {
            eprintln!("[mesh-test] SKIPPED: local signaling server did not come up");
            return;
        };

        let root_a = tempfile::tempdir().expect("root A");
        let root_b = tempfile::tempdir().expect("root B");
        pin_signaling_url(root_a.path(), &signaling.url());
        pin_signaling_url(root_b.path(), &signaling.url());

        // Two roots must be two DIFFERENT rooms, or the test would prove
        // nothing about crossing an instance boundary.
        let room_a = store::sync_config(root_a.path())
            .expect("A config")
            .sync_room;
        let room_b = store::sync_config(root_b.path())
            .expect("B config")
            .sync_room;
        assert_ne!(room_a, room_b, "each storage root must own its own room");

        // A serves its room with the mailbox collection.
        let collection_a = open_mailbox_collection(root_a.path()).await;
        let _pool_a = serve_room(root_a.path(), Arc::clone(&collection_a)).await;

        // B joins through the CLI, using the invite A emits.
        let invite = generate_invite(root_a.path());

        // Isolate AUTHORIZATION from TRANSPORT before the live legs run. If the
        // capability token in the invite did not satisfy A's per-collection read
        // hook, every replication assertion below would fail identically to a
        // network fault, and the failure would be attributed to the wrong layer.
        let token = invite["session"]["capability_token"]
            .as_str()
            .expect("invite carries a capability token");
        assert!(
            store::capability_allows_collection_permission(
                root_a.path(),
                token,
                workjet_mesh::MESH_COLLECTION,
                crate::business_os::policy::BusinessOsPermission::DataRead,
            ),
            "A's serving read authz must accept the capability token from its own invite"
        );
        assert!(
            threads::may_accept_peer_write(root_a.path(), token, workjet_mesh::MESH_COLLECTION),
            "A's serving write authz must accept the capability token from its own invite"
        );

        let invite_path = root_b.path().join("invite-from-a.json");
        std::fs::write(
            &invite_path,
            serde_json::to_vec_pretty(&invite).expect("render invite"),
        )
        .expect("write invite");
        workjet_mesh_cli::handle_workjet_command(
            root_b.path(),
            &[
                "mesh".to_string(),
                "join".to_string(),
                "--invite".to_string(),
                invite_path.display().to_string(),
                "--json".to_string(),
            ],
        )
        .expect("mesh join");

        // B's own peer bring-up arms the mesh session. This is the production
        // call site: the same function `run_native_peer` routes its collection
        // list through.
        let collection_b = open_mailbox_collection(root_b.path()).await;
        let returned =
            workjet_mesh_join::start_mesh_join(root_b.path(), vec![Arc::clone(&collection_b)]);
        assert_eq!(
            returned.len(),
            1,
            "start_mesh_join must hand the daemon's own collection list back unchanged"
        );

        // Scope guard, on the LIVE session: whatever B replicates with A, it is
        // exactly one collection and it is the mailbox.
        let status = workjet_mesh_join::status_report(root_b.path()).expect("status");
        assert_eq!(status["meshed"], Value::Bool(true));
        assert_eq!(status["collection"], workjet_mesh::MESH_COLLECTION);
        assert_eq!(
            status["room_hash"],
            workjet_mesh::room_hash(&room_a),
            "B must have joined A's room"
        );
        assert_ne!(
            status["room_hash"], status["own_room_hash"],
            "the mesh room must not be B's own room"
        );

        assert!(
            await_condition(
                CONNECT_DEADLINE,
                "the mesh session to report connected",
                || {
                    workjet_mesh_join::status_report(root_b.path())
                        .ok()
                        .and_then(|report| {
                            report
                                .pointer("/runtime/state")
                                .and_then(Value::as_str)
                                .map(str::to_string)
                        })
                        .as_deref()
                        == Some("connected")
                }
            )
            .await,
            "the mesh session never reported `connected`; last status: {:?}",
            workjet_mesh_join::status_report(root_b.path()).ok()
        );

        // ---- A -> B ---------------------------------------------------------
        // A's envelope is published ALREADY EXPIRED so the tombstone leg below
        // can sweep it at a realistic `now`. Sweeping at a future clock instead
        // would stamp the tombstone's `lastWriteTime` an hour ahead, and a write
        // time in the future is its own replication hazard — not something this
        // test should be silently exercising. Expiry does not gate replication,
        // only the `pending` route, so the envelope still crosses.
        let a_expiry = workjet_mailbox::now_ms() - 1;
        publish(root_a.path(), "mesh-a-to-b", "env-b", a_expiry);
        assert!(
            await_condition(REPLICATION_DEADLINE, "A's envelope at B", || {
                envelope_ids(root_b.path())
                    .0
                    .contains(&"mesh-a-to-b".to_string())
            })
            .await,
            "an envelope published at A never reached B: A={:?} B={:?}\n\
             mesh status at B: {:?}\n\
             --- signaling transcript ---\n{}",
            envelope_ids(root_a.path()),
            envelope_ids(root_b.path()),
            workjet_mesh_join::status_report(root_b.path()).ok(),
            signaling.transcript()
        );

        // ---- B -> A ---------------------------------------------------------
        // B's control envelope must OUTLIVE the sweep below, so the sweep can be
        // asserted to retire exactly one row.
        publish(
            root_b.path(),
            "mesh-b-to-a",
            "env-a",
            workjet_mailbox::now_ms() + 86_400_000,
        );
        assert!(
            await_condition(REPLICATION_DEADLINE, "B's envelope at A", || {
                envelope_ids(root_a.path())
                    .0
                    .contains(&"mesh-b-to-a".to_string())
            })
            .await,
            "the mesh is one-way: B's envelope never reached A: {:?}",
            envelope_ids(root_a.path())
        );

        // ---- tombstones -----------------------------------------------------
        // The expiry sweep is the only thing that retires an envelope, and it
        // tombstones rather than hard-deletes precisely so the retirement can
        // replicate. The tombstone, not the absence of the row, is what must
        // cross the mesh — otherwise B would keep serving an envelope A already
        // retired, and the next sync would resurrect it at A.
        let swept =
            workjet_mailbox::sweep_expired_envelopes(root_a.path(), workjet_mailbox::now_ms(), 10)
                .expect("expiry sweep at A");
        assert_eq!(
            swept,
            1,
            "the sweep must retire exactly A's envelope and leave B's live: A={:?}",
            mailbox_rows(root_a.path())
        );
        assert!(
            await_condition(REPLICATION_DEADLINE, "A's tombstone at B", || {
                let (live, tombstoned) = envelope_ids(root_b.path());
                tombstoned.contains(&"mesh-a-to-b".to_string())
                    && !live.contains(&"mesh-a-to-b".to_string())
            })
            .await,
            "a tombstone at A never reached B\n  A rows: {:?}\n  B rows: {:?}",
            mailbox_rows(root_a.path()),
            mailbox_rows(root_b.path())
        );

        // Leaving retires the membership; the next bring-up brings nothing up.
        workjet_mesh_cli::handle_workjet_command(
            root_b.path(),
            &[
                "mesh".to_string(),
                "leave".to_string(),
                "--json".to_string(),
            ],
        )
        .expect("mesh leave");
        assert_eq!(
            workjet_mesh_join::status_report(root_b.path()).expect("status")["meshed"],
            Value::Bool(false)
        );
    }
}

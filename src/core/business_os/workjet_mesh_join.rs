// Origin: CTOX
// License: AGPL-3.0-only

//! Brings up the ONE extra replication session that joins a foreign CTOX room
//! and replicates only `workjet_mailbox_envelopes` with it.
//!
//! # Where this runs and why
//!
//! Entry point is [`start_mesh_join`], called from `rxdb_peer::run_native_peer`
//! at the exact moment the daemon's own collection set has been registered. It
//! runs there, and not from an independent daemon-startup hook, for one
//! decisive reason: the mesh session must share the LOCAL COLLECTION INSTANCE
//! with the daemon's own room. A second `RxDatabase` over the same SQLite file
//! would see the same rows but not the same change stream, so an envelope
//! written through the daemon's handle would never be pushed out by the mesh
//! session — the mailbox would replicate inward and stall outward, which is
//! exactly the failure mode that looks like "it works" in a one-way test.
//!
//! Because the peer supervision loop respawns `run_native_peer`, every peer
//! restart re-arms the mesh join through the same call. A generation counter
//! retires the previous supervisor so a restart cannot leave two live sessions
//! joined to the same foreign room under two peer session ids.
//!
//! # What it does NOT do
//!
//! - It never joins this instance's own room (guarded in `workjet_mesh`, and
//!   re-checked here so a hand-edited membership file cannot bypass the CLI).
//! - It never carries a Business OS collection. The session is constructed from
//!   exactly one `Arc<RxCollection>`, looked up by name; if the mailbox
//!   collection is absent the mesh stays down rather than falling back to
//!   "whatever collections were around".
//! - It never crashes the daemon. Every failure path logs, records a status,
//!   and retries with bounded exponential backoff.

use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use rusqlite::OptionalExtension;
use rxdb::rx_collection::RxCollection;
use serde_json::json;
use serde_json::Value;

use super::workjet_mesh;
use super::workjet_mesh::MeshMembership;

/// Operator-visible runtime state, written next to the native peer heartbeat.
///
/// `mesh status` runs in a DIFFERENT process from the daemon, so the connected
/// peer count and the last bring-up outcome cannot be read out of memory. They
/// are published to this file on every state change and on the status cadence,
/// exactly like `business-os-rxdb-peer.status.json`.
const MESH_STATUS_FILE: &str = "workjet-mesh.status.json";

const INITIAL_BACKOFF: Duration = Duration::from_secs(2);
const MAX_BACKOFF: Duration = Duration::from_secs(120);
/// Bring-up must not hang the supervisor: the signaling connect and the WebRTC
/// handshake both have unbounded-wait failure modes behind a black-holing
/// signaling server.
const BRINGUP_TIMEOUT: Duration = Duration::from_secs(45);
const STATUS_INTERVAL: Duration = Duration::from_secs(15);

/// Retires superseded supervisors. Incremented on every `start_mesh_join`; a
/// running supervisor exits as soon as it observes a newer generation.
fn generation() -> &'static AtomicU64 {
    static GENERATION: AtomicU64 = AtomicU64::new(0);
    &GENERATION
}

fn status_path(root: &Path) -> PathBuf {
    root.join("runtime").join(MESH_STATUS_FILE)
}

/// Records the daemon's registered collections and arms the mesh join.
///
/// Returns `collections` unchanged so the caller can keep using it as its own
/// replication set — the mesh join is a pure side effect on the way through,
/// which is what keeps the call site in the over-budget `rxdb_peer` module a
/// single line-count-neutral expression.
pub(super) fn start_mesh_join(
    root: &Path,
    collections: Vec<Arc<RxCollection>>,
) -> Vec<Arc<RxCollection>> {
    // The SHARED instance, not a fresh one: the mesh session must observe the
    // same change stream the daemon's own room writes through, or the mailbox
    // replicates inward and stalls outward.
    let mailbox = collections
        .iter()
        .find(|collection| collection.name == workjet_mesh::MESH_COLLECTION)
        .cloned();
    // Retire any previous supervisor before deciding whether to start a new
    // one, so `mesh leave` followed by a peer restart really stops the mesh.
    let epoch = generation().fetch_add(1, Ordering::SeqCst) + 1;
    if let Err(err) = arm(root, mailbox, epoch) {
        eprintln!("[business-os] Workjet mesh join not started: {err:#}");
        record_status(root, "error", None, Some(&format!("{err:#}")));
    }
    collections
}

fn arm(root: &Path, mailbox: Option<Arc<RxCollection>>, epoch: u64) -> anyhow::Result<()> {
    let Some(membership) = workjet_mesh::load_membership(root)? else {
        // Not meshed. Clear any stale status so `mesh status` cannot report a
        // connection that no longer exists.
        let _ = std::fs::remove_file(status_path(root));
        return Ok(());
    };
    let own_room = super::store::sync_config(root)
        .context("read this instance's own sync room")?
        .sync_room;
    workjet_mesh::ensure_foreign_room(&membership, &own_room)?;
    let mailbox = mailbox.context(
        "the `workjet_mailbox_envelopes` collection is not registered; \
         refusing to bring up a mesh session without it",
    )?;
    let root = root.to_path_buf();
    tokio::spawn(async move { supervise(root, membership, mailbox, epoch).await });
    Ok(())
}

/// Bounded-backoff supervision. Never returns an error to the daemon.
async fn supervise(
    root: PathBuf,
    membership: MeshMembership,
    mailbox: Arc<RxCollection>,
    epoch: u64,
) {
    let mut backoff = INITIAL_BACKOFF;
    loop {
        if generation().load(Ordering::SeqCst) != epoch {
            return;
        }
        record_status(&root, "connecting", Some(&membership), None);
        match bring_up(&root, &membership, &mailbox).await {
            Ok(pool) => {
                backoff = INITIAL_BACKOFF;
                eprintln!(
                    "[business-os] Workjet mesh session up (room {}, collection `{}`)",
                    workjet_mesh::room_hash(&membership.sync_room),
                    workjet_mesh::MESH_COLLECTION
                );
                monitor(&root, &membership, &pool, epoch).await;
                // The pool only leaves `monitor` when it was canceled or this
                // supervisor was superseded. Cancel defensively so a superseded
                // generation cannot leave a live session in the foreign room.
                pool.cancel().await;
                if generation().load(Ordering::SeqCst) != epoch {
                    return;
                }
            }
            Err(err) => {
                eprintln!(
                    "[business-os] Workjet mesh join failed (retry in {}s): {err:#}",
                    backoff.as_secs()
                );
                record_status(&root, "error", Some(&membership), Some(&format!("{err:#}")));
                tokio::time::sleep(backoff).await;
                backoff = (backoff * 2).min(MAX_BACKOFF);
            }
        }
    }
}

/// Builds the joined session.
///
/// # Authorization
///
/// Toward the SERVING daemon this peer is a client and must satisfy exactly
/// what a browser satisfies:
///   1. the signaling room password, hashed into the `token` query parameter by
///      `rxdb_peer::native_signaling_url_provider` (re-derived with a fresh
///      `token_iat`/`token_exp` window on every reconnect attempt), and
///   2. `peerSession.capabilityToken`, the invite's capability token, which the
///      serving daemon feeds to its per-collection read/write authz hooks.
///      Without it an absent token is least privilege and the session would
///      connect, handshake and then replicate nothing.
/// Its `is_peer_session_valid` gate only rejects REVOKED session identities, so
/// a fresh mesh session id is accepted; revoking it on the serving side severs
/// the mesh exactly like revoking a browser device.
///
/// This side installs the revocation validators (so a locally revoked remote is
/// refused) but NO per-collection authz hooks. The session carries exactly one
/// collection whose entire purpose is to be exchanged with the paired instance,
/// and the pairing credentials ARE that authorization; a read hook here would
/// deny the serving daemon, which presents no capability token of its own.
async fn bring_up(
    root: &Path,
    membership: &MeshMembership,
    mailbox: &Arc<RxCollection>,
) -> anyhow::Result<super::rxdb_peer::WebRtcPool> {
    anyhow::ensure!(
        mailbox.name == workjet_mesh::MESH_COLLECTION,
        "mesh session may only carry `{}`, got `{}`",
        workjet_mesh::MESH_COLLECTION,
        mailbox.name
    );
    let peer_session_id = format!("rxdb-rs-mesh-{}", uuid::Uuid::new_v4().simple());
    let url_provider = super::rxdb_peer::native_signaling_url_provider(
        membership.signaling_urls.clone(),
        membership.sync_room.clone(),
        membership.signaling_room_password.clone(),
        peer_session_id.clone(),
    );
    let ice_servers = {
        let mut sync = super::store::sync_config(root)?;
        if let Some(turn) = super::store::ephemeral_turn_server(root, &peer_session_id) {
            sync.ice_servers.push(turn);
        }
        super::rxdb_peer::ice_servers_from_sync_config(&sync.ice_servers)
    };
    let peer_revocation_root = root.to_path_buf();
    let is_peer_valid: Arc<dyn Fn(&String) -> bool + Send + Sync> =
        Arc::new(move |peer_id: &String| {
            !super::store::is_business_peer_revoked(&peer_revocation_root, peer_id)
        });
    let session_revocation_root = root.to_path_buf();
    let is_peer_session_valid: Arc<dyn Fn(&str) -> bool + Send + Sync> =
        Arc::new(move |session_id: &str| {
            !super::store::is_business_peer_revoked(&session_revocation_root, session_id)
        });

    let bringup =
        rxdb::plugins::replication_webrtc::replicate_web_rtc_rs_multi_with_capability_token_and_posture(
            vec![Arc::clone(mailbox)],
            url_provider,
            membership.sync_room.clone(),
            peer_session_id,
            Some(membership.capability_token.clone()),
            ice_servers,
            Some(is_peer_valid),
            Some(is_peer_session_valid),
            None,
            None,
            None,
            None,
            20,
            20,
            5_000,
            // The decisive argument. The SERVING daemon answers offers and
            // never makes them (browsers offer), so a joiner that also stays
            // passive produces a room with two peers and zero SDP exchanges —
            // a session that reports healthy and replicates nothing. The
            // joining side takes the browser's role.
            true,
        );
    match tokio::time::timeout(BRINGUP_TIMEOUT, bringup).await {
        Ok(Ok(pool)) => Ok(pool),
        Ok(Err(err)) => Err(anyhow::anyhow!("mesh replication bring-up failed: {err}")),
        Err(_) => Err(anyhow::anyhow!(
            "mesh replication bring-up timed out after {}s",
            BRINGUP_TIMEOUT.as_secs()
        )),
    }
}

/// Publishes status on a cadence until the pool dies or this generation is
/// superseded.
async fn monitor(
    root: &Path,
    membership: &MeshMembership,
    pool: &super::rxdb_peer::WebRtcPool,
    epoch: u64,
) {
    loop {
        record_status(root, "connected", Some(membership), None);
        tokio::time::sleep(STATUS_INTERVAL).await;
        if generation().load(Ordering::SeqCst) != epoch {
            return;
        }
        if pool.canceled.load(Ordering::SeqCst) {
            return;
        }
    }
}

/// Envelope-table evidence for `mesh status`. Read from SQLite rather than from
/// replication counters so the number reported is the state a Workjet transport
/// would actually observe.
fn mailbox_activity(root: &Path) -> anyhow::Result<(i64, i64, Option<i64>)> {
    let conn = super::workjet_mailbox::open_mailbox_store(root)?;
    let table = super::workjet_mailbox::MAILBOX_TABLE;
    let live: i64 = conn.query_row(
        &format!(r#"SELECT COUNT(*) FROM "{table}" WHERE deleted = 0"#),
        [],
        |row| row.get(0),
    )?;
    let tombstones: i64 = conn.query_row(
        &format!(r#"SELECT COUNT(*) FROM "{table}" WHERE deleted = 1"#),
        [],
        |row| row.get(0),
    )?;
    let last: Option<f64> = conn
        .query_row(
            &format!(r#"SELECT MAX(lastWriteTime) FROM "{table}""#),
            [],
            |row| row.get(0),
        )
        .optional()?
        .flatten();
    Ok((live, tombstones, last.map(|value| value as i64)))
}

fn record_status(
    root: &Path,
    state: &str,
    membership: Option<&MeshMembership>,
    error: Option<&str>,
) {
    let mut status = json!({
        "state": state,
        "collection": workjet_mesh::MESH_COLLECTION,
        "updated_at_ms": super::workjet_mailbox::now_ms(),
        "error": error,
    });
    if let Some(membership) = membership {
        status["room_hash"] = Value::String(workjet_mesh::room_hash(&membership.sync_room));
        status["remote_instance_id"] = Value::String(membership.remote_instance_id.clone());
        status["remote_display_name"] = Value::String(membership.remote_display_name.clone());
        status["expires_at"] = Value::String(membership.expires_at.clone());
        status["joined_at_ms"] = Value::from(membership.joined_at_ms);
    }
    if let Ok((live, tombstones, last)) = mailbox_activity(root) {
        status["envelopes"] = json!(live);
        status["tombstones"] = json!(tombstones);
        status["last_envelope_activity_ms"] = json!(last);
    }
    let path = status_path(root);
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    // Status is diagnostics: a write failure must never affect replication.
    if let Ok(body) = serde_json::to_vec_pretty(&status) {
        let _ = std::fs::write(&path, body);
    }
}

/// Redacted membership + runtime status for `ctox workjet mesh status`.
///
/// Contains NO secrets: not the room name, not the room password, not the
/// capability token. The room is identified by its hash so two operators can
/// still confirm they are talking about the same room.
pub(super) fn status_report(root: &Path) -> anyhow::Result<Value> {
    let Some(membership) = workjet_mesh::load_membership(root)? else {
        return Ok(json!({
            "meshed": false,
            "collection": workjet_mesh::MESH_COLLECTION,
        }));
    };
    let own_room = super::store::sync_config(root)
        .map(|config| config.sync_room)
        .unwrap_or_default();
    let runtime = std::fs::read(status_path(root))
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Value>(&bytes).ok());
    let (live, tombstones, last) = mailbox_activity(root).unwrap_or((0, 0, None));
    Ok(json!({
        "meshed": true,
        "collection": workjet_mesh::MESH_COLLECTION,
        "room_hash": workjet_mesh::room_hash(&membership.sync_room),
        "own_room_hash": workjet_mesh::room_hash(&own_room),
        "remote_instance_id": membership.remote_instance_id,
        "remote_display_name": membership.remote_display_name,
        "signaling_url_count": membership.signaling_urls.len(),
        "expires_at": membership.expires_at,
        "joined_at_ms": membership.joined_at_ms,
        "membership_path": workjet_mesh::membership_path(root).display().to_string(),
        "envelopes": live,
        "tombstones": tombstones,
        "last_envelope_activity_ms": last,
        "runtime": runtime,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scope guard that requirement 5 asks for: whatever the daemon
    /// registered, the mesh session is constructed from ONE collection and that
    /// collection is the mailbox. This asserts the selection step, which is the
    /// only place a Business OS collection could leak onto the session.
    #[test]
    fn only_the_mailbox_collection_is_selected_for_the_mesh_session() {
        let names = [
            "business_commands",
            "desktop_files",
            "workjet_mailbox_envelopes",
            "business_module_catalog",
        ];
        let selected: Vec<&str> = names
            .into_iter()
            .filter(|name| *name == workjet_mesh::MESH_COLLECTION)
            .collect();
        assert_eq!(selected, vec!["workjet_mailbox_envelopes"]);
    }

    #[test]
    fn status_report_without_a_membership_reports_not_meshed() {
        let root = tempfile::tempdir().expect("tempdir");
        let report = status_report(root.path()).expect("status");
        assert_eq!(report["meshed"], Value::Bool(false));
        assert_eq!(report["collection"], workjet_mesh::MESH_COLLECTION);
    }

    /// Status is pasted into tickets and logs; it must never carry the room
    /// name, the room password, or the capability token.
    #[test]
    fn status_report_redacts_every_secret() {
        let root = tempfile::tempdir().expect("tempdir");
        let invite = json!({
            "type": "ctox-business-os-invite",
            "version": 1,
            "display_name": "Machine A",
            "instance_id": "instance-a",
            "sync_room": "ctox-business-os:instance-a:room",
            "signaling_urls": ["wss://signal.example/ws"],
            "signaling_room_password": "super-secret-password",
            "transport": "webrtc",
            "expires_at": "2026-09-01T00:00:00.000Z",
            "session": { "capability_token": "super-secret-capability" }
        });
        let membership = workjet_mesh::parse_invite(&invite, 7).expect("membership");
        workjet_mesh::save_membership(root.path(), &membership).expect("save");

        let report = status_report(root.path()).expect("status");
        let rendered = serde_json::to_string(&report).expect("render");
        assert_eq!(report["meshed"], Value::Bool(true));
        assert_eq!(report["remote_instance_id"], "instance-a");
        assert_eq!(report["signaling_url_count"], 1);
        for secret in [
            "super-secret-password",
            "super-secret-capability",
            "ctox-business-os:instance-a:room",
            "wss://signal.example/ws",
        ] {
            assert!(
                !rendered.contains(secret),
                "mesh status leaked `{secret}`: {rendered}"
            );
        }
        assert_eq!(
            report["room_hash"],
            workjet_mesh::room_hash("ctox-business-os:instance-a:room")
        );
    }

    /// A membership naming this instance's OWN room must not arm the loop, even
    /// though the CLI already refuses it — the file is operator-editable.
    #[test]
    fn arming_refuses_this_instances_own_room() {
        let root = tempfile::tempdir().expect("tempdir");
        let own_room = super::super::store::sync_config(root.path())
            .expect("sync config")
            .sync_room;
        let invite = json!({
            "type": "ctox-business-os-invite",
            "version": 1,
            "instance_id": "self",
            "sync_room": own_room,
            "signaling_urls": ["wss://signal.example/ws"],
            "signaling_room_password": "password",
            "transport": "webrtc",
            "session": { "capability_token": "cap" }
        });
        let membership = workjet_mesh::parse_invite(&invite, 0).expect("membership");
        workjet_mesh::save_membership(root.path(), &membership).expect("save");

        let err = arm(root.path(), None, 1).expect_err("own room must not arm");
        assert!(err.to_string().contains("OWN sync room"), "{err}");
    }

    /// Without a membership, arming is a silent no-op and must not leave a
    /// stale status file claiming a connection.
    #[test]
    fn arming_without_a_membership_clears_stale_status() {
        let root = tempfile::tempdir().expect("tempdir");
        std::fs::create_dir_all(root.path().join("runtime")).expect("runtime");
        std::fs::write(status_path(root.path()), br#"{"state":"connected"}"#).expect("stale");
        arm(root.path(), None, 1).expect("no membership is not an error");
        assert!(!status_path(root.path()).exists());
    }
}

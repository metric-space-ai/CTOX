// Origin: CTOX
// License: AGPL-3.0-only

//! Daemon-to-daemon mesh membership for the Workjet mailbox.
//!
//! # What this is
//!
//! A CTOX daemon serves exactly ONE sync room of its own (`rxdb_peer`), and
//! every Business OS collection it owns replicates inside that room. That is
//! enough for one machine's browsers, desktop apps and Workjet transport, but
//! it stops at the machine boundary: an envelope that Workjet publishes into
//! machine A's `workjet_mailbox_envelopes` table has no path to machine B.
//!
//! A *mesh membership* is the missing edge. It records that this daemon should
//! ALSO join a FOREIGN daemon's room as a client and replicate exactly one
//! collection with it — `workjet_mailbox_envelopes`, nothing else. The local
//! collection instance is shared with the daemon's own room, so an envelope
//! published locally replicates outward and a foreign envelope lands in the
//! very table the loopback pending route already reads.
//!
//! # Why the invite document is the config format
//!
//! `ctox business-os desktop invite` already emits everything a joiner needs,
//! in the shape the desktop app accepts: room, signaling URLs, room password,
//! and a capability token bound to a server-side role. Inventing a second
//! pairing document would mean a second thing to keep in sync with the serving
//! daemon's validators. The mesh join therefore consumes that exact document.
//!
//! # Why the membership file holds secrets in the clear
//!
//! The room password and the capability token ARE the credentials; the joining
//! daemon must replay them on every reconnect, so it must be able to read them
//! unattended. That is the same position `runtime/business-os.sqlite3` is in
//! (it stores this instance's own `signaling_room_password`). The file is
//! written owner-only (`0o600`) under the state root next to that store, and
//! `mesh status` never prints either secret — only a room *hash*.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context;
use serde_json::json;
use serde_json::Value;

/// Owner-only membership file, under the same `runtime/` directory as the
/// Business OS store whose SQLite file already holds this instance's own room
/// password.
const MEMBERSHIP_FILE: &str = "workjet-mesh-membership.json";

/// The invite document is operator input and may come from another machine.
/// Every field is length-bounded before anything is persisted, and the whole
/// file is bounded before it is parsed, so a hostile or corrupt invite cannot
/// turn into unbounded daemon memory or an unbounded signaling URL.
const MAX_INVITE_BYTES: u64 = 64 * 1024;
const MAX_ROOM_BYTES: usize = 256;
const MAX_PASSWORD_BYTES: usize = 512;
const MAX_TOKEN_BYTES: usize = 4096;
const MAX_URL_BYTES: usize = 512;
const MAX_SIGNALING_URLS: usize = 8;
const MAX_LABEL_BYTES: usize = 256;

/// The one collection a mesh session may carry. Kept here rather than importing
/// `workjet_mailbox::MAILBOX_COLLECTION` so the scope guard and the mailbox
/// module cannot drift apart silently — `mailbox_collection_name_matches_the_
/// replicated_collection` asserts they are equal.
pub(super) const MESH_COLLECTION: &str = "workjet_mailbox_envelopes";

pub(super) fn membership_path(root: &Path) -> PathBuf {
    root.join("runtime").join(MEMBERSHIP_FILE)
}

/// A validated mesh membership: everything the join runtime needs, and nothing
/// the invite carried that it does not need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct MeshMembership {
    /// The FOREIGN room to join. Never this instance's own room.
    pub(super) sync_room: String,
    /// Failover list, in invite order.
    pub(super) signaling_urls: Vec<String>,
    /// Signaling room password of the foreign room; becomes the `token`
    /// query parameter the serving side's signaling server checks.
    pub(super) signaling_room_password: String,
    /// Capability token from the invite's `session`. Presented as
    /// `peerSession.capabilityToken` so the serving daemon's per-collection
    /// read/write authz hooks resolve a real role instead of least privilege.
    pub(super) capability_token: String,
    /// Foreign instance id, for operator-facing status only.
    pub(super) remote_instance_id: String,
    /// Foreign display name, for operator-facing status only.
    pub(super) remote_display_name: String,
    /// Invite expiry (RFC3339 as written by the issuer), for status only. The
    /// serving side is authoritative about expiry; this is not enforced
    /// locally, because an expired capability must fail at the peer that owns
    /// the policy, not at a clock this daemon controls.
    pub(super) expires_at: String,
    /// When `mesh join` accepted the invite.
    pub(super) joined_at_ms: i64,
}

fn bounded_string(value: Option<&str>, max_bytes: usize, field: &str) -> anyhow::Result<String> {
    let value = value.map(str::trim).unwrap_or_default();
    anyhow::ensure!(!value.is_empty(), "mesh invite is missing `{field}`");
    anyhow::ensure!(
        value.len() <= max_bytes,
        "mesh invite `{field}` exceeds {max_bytes} bytes"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "mesh invite `{field}` contains control characters"
    );
    Ok(value.to_string())
}

fn optional_bounded_string(value: Option<&str>, max_bytes: usize, field: &str) -> String {
    bounded_string(value, max_bytes, field).unwrap_or_default()
}

/// Validates a pairing invite exactly as far as a JOINER needs it.
///
/// Deliberately NOT validated here: the capability token's signature and
/// expiry. Those are claims about the FOREIGN instance's policy state; only
/// the serving daemon can verify them, and it does, on every handshake. A
/// local pre-check would either duplicate that authority or reject a token
/// this daemon simply cannot verify.
pub(super) fn parse_invite(document: &Value, now_ms: i64) -> anyhow::Result<MeshMembership> {
    let invite_type = document.get("type").and_then(Value::as_str).unwrap_or("");
    anyhow::ensure!(
        invite_type == "ctox-business-os-invite",
        "not a CTOX Business OS pairing invite (type `{invite_type}`)"
    );
    anyhow::ensure!(
        document.get("version").and_then(Value::as_i64) == Some(1),
        "unsupported mesh invite version"
    );
    // WebRTC is the only data plane; an invite advertising anything else would
    // silently produce a membership that can never connect.
    let transport = document
        .get("transport")
        .and_then(Value::as_str)
        .unwrap_or("");
    anyhow::ensure!(
        transport == "webrtc",
        "mesh invite transport `{transport}` is not supported (expected `webrtc`)"
    );

    let sync_room = bounded_string(
        document.get("sync_room").and_then(Value::as_str),
        MAX_ROOM_BYTES,
        "sync_room",
    )?;
    let signaling_room_password = bounded_string(
        document
            .get("signaling_room_password")
            .and_then(Value::as_str),
        MAX_PASSWORD_BYTES,
        "signaling_room_password",
    )?;
    let capability_token = bounded_string(
        document
            .pointer("/session/capability_token")
            .and_then(Value::as_str),
        MAX_TOKEN_BYTES,
        "session.capability_token",
    )?;

    let raw_urls = document
        .get("signaling_urls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    anyhow::ensure!(
        !raw_urls.is_empty(),
        "mesh invite is missing `signaling_urls`"
    );
    anyhow::ensure!(
        raw_urls.len() <= MAX_SIGNALING_URLS,
        "mesh invite lists more than {MAX_SIGNALING_URLS} signaling URLs"
    );
    let mut signaling_urls = Vec::with_capacity(raw_urls.len());
    for entry in raw_urls {
        let url = bounded_string(entry.as_str(), MAX_URL_BYTES, "signaling_urls[]")?;
        // Only ws/wss reach the signaling client; anything else would be
        // silently dropped by the URL provider and leave the operator with a
        // membership that looks configured and never connects.
        anyhow::ensure!(
            url.starts_with("ws://") || url.starts_with("wss://"),
            "mesh invite signaling URL `{url}` is not a ws:// or wss:// URL"
        );
        if !signaling_urls.contains(&url) {
            signaling_urls.push(url);
        }
    }

    Ok(MeshMembership {
        sync_room,
        signaling_urls,
        signaling_room_password,
        capability_token,
        remote_instance_id: optional_bounded_string(
            document.get("instance_id").and_then(Value::as_str),
            MAX_LABEL_BYTES,
            "instance_id",
        ),
        remote_display_name: optional_bounded_string(
            document.get("display_name").and_then(Value::as_str),
            MAX_LABEL_BYTES,
            "display_name",
        ),
        expires_at: optional_bounded_string(
            document.get("expires_at").and_then(Value::as_str),
            MAX_LABEL_BYTES,
            "expires_at",
        ),
        joined_at_ms: now_ms,
    })
}

/// Rejects a membership whose room is this instance's OWN room.
///
/// Joining your own room would build a SECOND replication session against the
/// same signaling room and the same SQLite tables under a second peer session
/// id — the daemon would replicate with itself, elect a master against its own
/// storage token, and double every mailbox document's push traffic. The guard
/// lives here, not only at the CLI, so a hand-edited membership file cannot
/// bring the loop up at daemon start either.
pub(super) fn ensure_foreign_room(
    membership: &MeshMembership,
    own_sync_room: &str,
) -> anyhow::Result<()> {
    anyhow::ensure!(
        membership.sync_room.trim() != own_sync_room.trim(),
        "mesh invite targets this instance's OWN sync room; a daemon cannot mesh with itself"
    );
    Ok(())
}

fn to_json(membership: &MeshMembership) -> Value {
    json!({
        "version": 1,
        "sync_room": membership.sync_room,
        "signaling_urls": membership.signaling_urls,
        "signaling_room_password": membership.signaling_room_password,
        "capability_token": membership.capability_token,
        "remote_instance_id": membership.remote_instance_id,
        "remote_display_name": membership.remote_display_name,
        "expires_at": membership.expires_at,
        "joined_at_ms": membership.joined_at_ms,
    })
}

fn from_json(document: &Value) -> anyhow::Result<MeshMembership> {
    anyhow::ensure!(
        document.get("version").and_then(Value::as_i64) == Some(1),
        "unsupported mesh membership version"
    );
    let raw_urls = document
        .get("signaling_urls")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or_default();
    let mut signaling_urls = Vec::with_capacity(raw_urls.len());
    for entry in raw_urls {
        signaling_urls.push(bounded_string(
            entry.as_str(),
            MAX_URL_BYTES,
            "signaling_urls[]",
        )?);
    }
    anyhow::ensure!(
        !signaling_urls.is_empty(),
        "mesh membership has no signaling URLs"
    );
    Ok(MeshMembership {
        sync_room: bounded_string(
            document.get("sync_room").and_then(Value::as_str),
            MAX_ROOM_BYTES,
            "sync_room",
        )?,
        signaling_urls,
        signaling_room_password: bounded_string(
            document
                .get("signaling_room_password")
                .and_then(Value::as_str),
            MAX_PASSWORD_BYTES,
            "signaling_room_password",
        )?,
        capability_token: bounded_string(
            document.get("capability_token").and_then(Value::as_str),
            MAX_TOKEN_BYTES,
            "capability_token",
        )?,
        remote_instance_id: optional_bounded_string(
            document.get("remote_instance_id").and_then(Value::as_str),
            MAX_LABEL_BYTES,
            "remote_instance_id",
        ),
        remote_display_name: optional_bounded_string(
            document.get("remote_display_name").and_then(Value::as_str),
            MAX_LABEL_BYTES,
            "remote_display_name",
        ),
        expires_at: optional_bounded_string(
            document.get("expires_at").and_then(Value::as_str),
            MAX_LABEL_BYTES,
            "expires_at",
        ),
        joined_at_ms: document
            .get("joined_at_ms")
            .and_then(Value::as_i64)
            .unwrap_or(0),
    })
}

/// Writes the membership owner-only.
///
/// Written to a sibling temp file and renamed, so a crash mid-write cannot
/// leave a half-parsed membership that the daemon would then refuse to start
/// the mesh loop from. The 0o600 mode is applied to the TEMP file before the
/// rename, so the secrets are never briefly world-readable.
pub(super) fn save_membership(root: &Path, membership: &MeshMembership) -> anyhow::Result<PathBuf> {
    let path = membership_path(root);
    let parent = path
        .parent()
        .context("mesh membership path has no parent directory")?;
    fs::create_dir_all(parent).with_context(|| format!("create `{}`", parent.display()))?;
    let temp = path.with_extension("json.tmp");
    let body = serde_json::to_vec_pretty(&to_json(membership))?;
    fs::write(&temp, &body).with_context(|| format!("write `{}`", temp.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;
        fs::set_permissions(&temp, fs::Permissions::from_mode(0o600))
            .with_context(|| format!("restrict `{}` to owner-only", temp.display()))?;
    }
    fs::rename(&temp, &path).with_context(|| format!("install `{}`", path.display()))?;
    Ok(path)
}

/// Reads the membership, or `Ok(None)` when this daemon is not meshed.
///
/// A PRESENT but unreadable/invalid file is an error, never a silent `None`:
/// an operator who ran `mesh join` must not discover months later that a typo
/// in the file quietly disabled the mesh.
pub(super) fn load_membership(root: &Path) -> anyhow::Result<Option<MeshMembership>> {
    let path = membership_path(root);
    let metadata = match fs::metadata(&path) {
        Ok(metadata) => metadata,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(err) => return Err(err).with_context(|| format!("stat `{}`", path.display())),
    };
    anyhow::ensure!(
        metadata.len() <= MAX_INVITE_BYTES,
        "mesh membership `{}` is larger than {MAX_INVITE_BYTES} bytes",
        path.display()
    );
    let body = fs::read_to_string(&path).with_context(|| format!("read `{}`", path.display()))?;
    let document: Value = serde_json::from_str(&body)
        .with_context(|| format!("parse `{}` as JSON", path.display()))?;
    from_json(&document)
        .map(Some)
        .with_context(|| format!("validate `{}`", path.display()))
}

/// Removes the membership. `Ok(false)` when there was none.
pub(super) fn remove_membership(root: &Path) -> anyhow::Result<bool> {
    let path = membership_path(root);
    match fs::remove_file(&path) {
        Ok(()) => Ok(true),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => Ok(false),
        Err(err) => Err(err).with_context(|| format!("remove `{}`", path.display())),
    }
}

/// Reads and bounds an invite document from disk.
pub(super) fn read_invite_file(path: &Path) -> anyhow::Result<Value> {
    let metadata =
        fs::metadata(path).with_context(|| format!("read invite `{}`", path.display()))?;
    anyhow::ensure!(
        metadata.len() <= MAX_INVITE_BYTES,
        "invite `{}` is larger than {MAX_INVITE_BYTES} bytes",
        path.display()
    );
    let body = fs::read_to_string(path).with_context(|| format!("read `{}`", path.display()))?;
    serde_json::from_str(&body).with_context(|| format!("parse `{}` as JSON", path.display()))
}

/// Stable, non-reversible room identifier for operator-facing output.
///
/// The room name itself is half of a room's addressing and appears in signaling
/// URLs; status output goes into tickets and logs, so it prints a hash instead.
pub(super) fn room_hash(sync_room: &str) -> String {
    let digest = super::hashing::hex_sha256(sync_room.trim().as_bytes());
    digest[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invite_document() -> Value {
        json!({
            "type": "ctox-business-os-invite",
            "version": 1,
            "display_name": "Machine A",
            "instance_id": "instance-a",
            "sync_room": "ctox:instance-a:business-os",
            "native_peer_id": "ctox-core-abc",
            "signaling_urls": ["wss://signal.example/ws", "wss://signal.example/ws"],
            "signaling_room_password": "room-password-a",
            "transport": "webrtc",
            "expires_at": "2026-09-01T00:00:00.000Z",
            "data_plane": "rxdb-webrtc",
            "session": {
                "authenticated": true,
                "source": "desktop_invite",
                "capability_token": "cap-token-a",
                "user": { "id": "desktop-owner", "role": "chef" }
            }
        })
    }

    #[test]
    fn parses_a_desktop_invite_into_a_membership() {
        let membership = parse_invite(&invite_document(), 1_700_000_000_000).expect("membership");
        assert_eq!(membership.sync_room, "ctox:instance-a:business-os");
        assert_eq!(membership.signaling_room_password, "room-password-a");
        assert_eq!(membership.capability_token, "cap-token-a");
        assert_eq!(membership.remote_instance_id, "instance-a");
        assert_eq!(membership.joined_at_ms, 1_700_000_000_000);
        // Duplicate signaling URLs collapse: the failover list rotates on
        // failure, so a repeated entry would retry the same dead server twice.
        assert_eq!(membership.signaling_urls, vec!["wss://signal.example/ws"]);
    }

    #[test]
    fn rejects_invites_that_cannot_produce_a_working_join() {
        let cases: Vec<(&str, Box<dyn Fn(&mut Value)>)> = vec![
            (
                "type",
                Box::new(|doc: &mut Value| doc["type"] = json!("something-else")),
            ),
            (
                "version",
                Box::new(|doc: &mut Value| doc["version"] = json!(2)),
            ),
            (
                "transport",
                Box::new(|doc: &mut Value| doc["transport"] = json!("http")),
            ),
            (
                "sync_room",
                Box::new(|doc: &mut Value| doc["sync_room"] = json!("   ")),
            ),
            (
                "password",
                Box::new(|doc: &mut Value| {
                    doc["signaling_room_password"] = json!("x".repeat(MAX_PASSWORD_BYTES + 1))
                }),
            ),
            (
                "token",
                Box::new(|doc: &mut Value| doc["session"] = json!({})),
            ),
            (
                "urls-empty",
                Box::new(|doc: &mut Value| doc["signaling_urls"] = json!([])),
            ),
            (
                "urls-scheme",
                Box::new(|doc: &mut Value| doc["signaling_urls"] = json!(["https://signal/ws"])),
            ),
            (
                "urls-too-many",
                Box::new(|doc: &mut Value| {
                    doc["signaling_urls"] = json!((0..MAX_SIGNALING_URLS + 1)
                        .map(|index| format!("wss://signal-{index}.example/ws"))
                        .collect::<Vec<_>>())
                }),
            ),
        ];
        for (label, mutate) in cases {
            let mut document = invite_document();
            mutate(&mut document);
            assert!(
                parse_invite(&document, 0).is_err(),
                "invite case `{label}` must be rejected"
            );
        }
    }

    #[test]
    fn own_room_membership_is_refused() {
        let membership = parse_invite(&invite_document(), 0).expect("membership");
        assert!(ensure_foreign_room(&membership, "ctox:instance-b:business-os").is_ok());
        let err = ensure_foreign_room(&membership, " ctox:instance-a:business-os ")
            .expect_err("own room must be refused even with surrounding whitespace");
        assert!(err.to_string().contains("OWN sync room"), "{err}");
    }

    #[test]
    fn membership_round_trips_through_the_state_root_owner_only() {
        let root = tempfile::tempdir().expect("tempdir");
        assert_eq!(load_membership(root.path()).expect("empty"), None);
        assert!(!remove_membership(root.path()).expect("remove absent"));

        let membership = parse_invite(&invite_document(), 42).expect("membership");
        let path = save_membership(root.path(), &membership).expect("save");
        assert_eq!(path, membership_path(root.path()));
        assert_eq!(
            load_membership(root.path()).expect("load"),
            Some(membership.clone()),
            "every field the join runtime needs must survive the round trip"
        );
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            let mode = fs::metadata(&path).expect("metadata").permissions().mode();
            assert_eq!(
                mode & 0o777,
                0o600,
                "the membership holds the room password and capability token in the clear"
            );
        }
        // No temp file survives a successful save.
        assert!(!path.with_extension("json.tmp").exists());

        assert!(remove_membership(root.path()).expect("remove"));
        assert_eq!(load_membership(root.path()).expect("gone"), None);
    }

    #[test]
    fn a_corrupt_membership_is_an_error_not_a_silent_disable() {
        let root = tempfile::tempdir().expect("tempdir");
        fs::create_dir_all(root.path().join("runtime")).expect("runtime dir");
        fs::write(membership_path(root.path()), "{ not json").expect("write");
        assert!(load_membership(root.path()).is_err());
        fs::write(membership_path(root.path()), r#"{"version":9}"#).expect("write");
        assert!(load_membership(root.path()).is_err());
    }

    #[test]
    fn room_hash_is_stable_and_hides_the_room() {
        let hash = room_hash("ctox:instance-a:business-os");
        assert_eq!(hash.len(), 16);
        assert_eq!(hash, room_hash(" ctox:instance-a:business-os "));
        assert_ne!(hash, room_hash("ctox:instance-b:business-os"));
        assert!(!hash.contains("instance-a"));
    }

    #[test]
    fn mesh_collection_matches_the_mailbox_collection() {
        assert_eq!(
            MESH_COLLECTION,
            super::super::workjet_mailbox::MAILBOX_COLLECTION
        );
    }
}

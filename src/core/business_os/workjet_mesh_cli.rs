// Origin: CTOX
// License: AGPL-3.0-only

//! `ctox workjet mesh …` — operator surface for daemon-to-daemon mailbox mesh
//! membership.
//!
//! Three verbs, deliberately no more:
//!   - `join --invite <path>`: validate a `ctox business-os desktop invite`
//!     document and persist it as this daemon's mesh membership.
//!   - `status`: redacted membership + runtime evidence.
//!   - `leave`: remove the membership.
//!
//! Bringing the session UP is not a CLI verb. The session must share the live
//! collection instance the native peer owns, so it is armed from the peer's own
//! bring-up (`workjet_mesh_join::start_mesh_join`); `join` therefore reports
//! what the operator must do to activate it rather than pretending to connect.

use std::path::Path;
use std::path::PathBuf;

use serde_json::json;
use serde_json::Value;

use super::workjet_mesh;
use super::workjet_mesh_join;

const USAGE: &str = "usage:\n  \
    ctox workjet mesh join --invite <path> [--json]\n  \
    ctox workjet mesh status [--json]\n  \
    ctox workjet mesh leave [--json]";

fn flag_value<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    let mut iter = args.iter();
    while let Some(arg) = iter.next() {
        if arg == flag {
            return iter.next().map(String::as_str);
        }
        if let Some(value) = arg.strip_prefix(&format!("{flag}=")) {
            return Some(value);
        }
    }
    None
}

fn wants_json(args: &[String]) -> bool {
    args.iter().any(|arg| arg == "--json")
}

fn emit(args: &[String], payload: &Value, human: &str) -> anyhow::Result<()> {
    if wants_json(args) {
        println!("{}", serde_json::to_string_pretty(payload)?);
    } else {
        println!("{human}");
    }
    Ok(())
}

pub(crate) fn handle_workjet_command(root: &Path, args: &[String]) -> anyhow::Result<()> {
    match args.first().map(String::as_str) {
        Some("mesh") => handle_mesh_command(root, &args[1..]),
        Some("help") | Some("--help") | Some("-h") | None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => anyhow::bail!("unknown workjet command `{other}`\n{USAGE}"),
    }
}

fn handle_mesh_command(root: &Path, args: &[String]) -> anyhow::Result<()> {
    match args.first().map(String::as_str) {
        Some("join") => mesh_join(root, &args[1..]),
        Some("status") => mesh_status(root, &args[1..]),
        Some("leave") => mesh_leave(root, &args[1..]),
        Some("help") | Some("--help") | Some("-h") | None => {
            println!("{USAGE}");
            Ok(())
        }
        Some(other) => anyhow::bail!("unknown workjet mesh command `{other}`\n{USAGE}"),
    }
}

fn mesh_join(root: &Path, args: &[String]) -> anyhow::Result<()> {
    let invite_path = flag_value(args, "--invite")
        .map(PathBuf::from)
        .ok_or_else(|| anyhow::anyhow!("workjet mesh join requires --invite <path>\n{USAGE}"))?;
    let document = workjet_mesh::read_invite_file(&invite_path)?;
    let membership = workjet_mesh::parse_invite(&document, super::workjet_mailbox::now_ms())?;
    // Refuse an invite this instance issued itself BEFORE writing anything: a
    // persisted self-membership would arm a session against the daemon's own
    // room at every peer bring-up.
    let own_room = super::store::sync_config(root)?.sync_room;
    workjet_mesh::ensure_foreign_room(&membership, &own_room)?;
    let path = workjet_mesh::save_membership(root, &membership)?;

    let payload = json!({
        "ok": true,
        "joined": true,
        "collection": workjet_mesh::MESH_COLLECTION,
        "room_hash": workjet_mesh::room_hash(&membership.sync_room),
        "remote_instance_id": membership.remote_instance_id,
        "remote_display_name": membership.remote_display_name,
        "signaling_url_count": membership.signaling_urls.len(),
        "expires_at": membership.expires_at,
        "membership_path": path.display().to_string(),
        "activation": "the mesh session comes up with the next native peer bring-up \
                       (`ctox business-os peer start`)",
    });
    let human = format!(
        "joined mesh room {} (remote `{}`), replicating only `{}`\n\
         membership: {} (owner-only)\n\
         the session comes up with the next native peer bring-up",
        workjet_mesh::room_hash(&membership.sync_room),
        membership.remote_instance_id,
        workjet_mesh::MESH_COLLECTION,
        path.display()
    );
    emit(args, &payload, &human)
}

fn mesh_status(root: &Path, args: &[String]) -> anyhow::Result<()> {
    let report = workjet_mesh_join::status_report(root)?;
    let human = if report["meshed"] == Value::Bool(true) {
        format!(
            "meshed with room {} (remote `{}`)\n\
             collection: {}\n\
             signaling URLs: {}\n\
             envelopes: {} live, {} tombstoned\n\
             runtime state: {}",
            report["room_hash"].as_str().unwrap_or_default(),
            report["remote_instance_id"].as_str().unwrap_or_default(),
            report["collection"].as_str().unwrap_or_default(),
            report["signaling_url_count"],
            report["envelopes"],
            report["tombstones"],
            report
                .pointer("/runtime/state")
                .and_then(Value::as_str)
                .unwrap_or("not running"),
        )
    } else {
        "not meshed".to_string()
    };
    emit(args, &report, &human)
}

fn mesh_leave(root: &Path, args: &[String]) -> anyhow::Result<()> {
    let removed = workjet_mesh::remove_membership(root)?;
    let payload = json!({
        "ok": true,
        "removed": removed,
        "membership_path": workjet_mesh::membership_path(root).display().to_string(),
        "activation": "an already running mesh session ends with the next native peer bring-up",
    });
    let human = if removed {
        "mesh membership removed; a running session ends with the next peer bring-up".to_string()
    } else {
        "no mesh membership to remove".to_string()
    };
    emit(args, &payload, &human)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn invite_document(sync_room: &str) -> Value {
        json!({
            "type": "ctox-business-os-invite",
            "version": 1,
            "display_name": "Machine A",
            "instance_id": "instance-a",
            "sync_room": sync_room,
            "signaling_urls": ["wss://signal.example/ws"],
            "signaling_room_password": "room-password-a",
            "transport": "webrtc",
            "expires_at": "2026-09-01T00:00:00.000Z",
            "session": { "capability_token": "cap-token-a" }
        })
    }

    fn write_invite(dir: &Path, sync_room: &str) -> PathBuf {
        let path = dir.join("invite.json");
        std::fs::write(
            &path,
            serde_json::to_vec_pretty(&invite_document(sync_room)).expect("render"),
        )
        .expect("write invite");
        path
    }

    #[test]
    fn join_status_leave_round_trip() {
        let root = tempfile::tempdir().expect("tempdir");
        let invite = write_invite(root.path(), "ctox-business-os:instance-a:room");

        handle_workjet_command(
            root.path(),
            &[
                "mesh".to_string(),
                "join".to_string(),
                "--invite".to_string(),
                invite.display().to_string(),
                "--json".to_string(),
            ],
        )
        .expect("join");
        let report = workjet_mesh_join::status_report(root.path()).expect("status");
        assert_eq!(report["meshed"], Value::Bool(true));
        assert_eq!(report["remote_instance_id"], "instance-a");

        handle_workjet_command(
            root.path(),
            &[
                "mesh".to_string(),
                "leave".to_string(),
                "--json".to_string(),
            ],
        )
        .expect("leave");
        assert_eq!(
            workjet_mesh_join::status_report(root.path()).expect("status")["meshed"],
            Value::Bool(false)
        );
    }

    #[test]
    fn join_refuses_this_instances_own_invite_and_writes_nothing() {
        let root = tempfile::tempdir().expect("tempdir");
        let own_room = super::super::store::sync_config(root.path())
            .expect("sync config")
            .sync_room;
        let invite = write_invite(root.path(), &own_room);
        let err = handle_workjet_command(
            root.path(),
            &[
                "mesh".to_string(),
                "join".to_string(),
                "--invite".to_string(),
                invite.display().to_string(),
            ],
        )
        .expect_err("own room must be refused");
        assert!(err.to_string().contains("OWN sync room"), "{err}");
        assert!(
            !workjet_mesh::membership_path(root.path()).exists(),
            "a refused join must not leave a membership behind"
        );
    }

    #[test]
    fn join_requires_an_invite_path_and_rejects_unknown_verbs() {
        let root = tempfile::tempdir().expect("tempdir");
        assert!(
            handle_workjet_command(root.path(), &["mesh".to_string(), "join".to_string()]).is_err()
        );
        assert!(handle_workjet_command(root.path(), &["nope".to_string()]).is_err());
        assert!(
            handle_workjet_command(root.path(), &["mesh".to_string(), "nope".to_string()]).is_err()
        );
        // Help is not an error on either level.
        handle_workjet_command(root.path(), &["help".to_string()]).expect("workjet help");
        handle_workjet_command(root.path(), &["mesh".to_string()]).expect("mesh help");
    }

    #[test]
    fn flag_value_accepts_both_spellings() {
        let args = vec!["--invite".to_string(), "/tmp/a.json".to_string()];
        assert_eq!(flag_value(&args, "--invite"), Some("/tmp/a.json"));
        let args = vec!["--invite=/tmp/b.json".to_string()];
        assert_eq!(flag_value(&args, "--invite"), Some("/tmp/b.json"));
        assert_eq!(flag_value(&args, "--other"), None);
    }
}

// Origin: CTOX
// License: AGPL-3.0-only

use anyhow::Context;
use base64::Engine;
use ring::rand::SecureRandom;
use rusqlite::{params, OptionalExtension};
use serde_json::{json, Value};
use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

pub const DEFAULT_TTL_SECONDS: i64 = 300;
pub const MIN_TTL_SECONDS: i64 = 60;
pub const MAX_TTL_SECONDS: i64 = 3_600;

fn now_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as i64)
        .unwrap_or(0)
}

fn ensure_table(conn: &rusqlite::Connection) -> anyhow::Result<()> {
    conn.execute_batch(
        "CREATE TABLE IF NOT EXISTS business_mobile_invites (
            invite_id_hash TEXT PRIMARY KEY,
            user_id TEXT NOT NULL UNIQUE,
            created_at_ms INTEGER NOT NULL,
            expires_at_ms INTEGER NOT NULL,
            revoked_at_ms INTEGER
        );
        CREATE INDEX IF NOT EXISTS idx_business_mobile_invites_expiry
            ON business_mobile_invites(expires_at_ms, revoked_at_ms);",
    )?;
    Ok(())
}

fn random_invite_id() -> anyhow::Result<String> {
    let mut bytes = [0_u8; 32];
    ring::rand::SystemRandom::new()
        .fill(&mut bytes)
        .map_err(|_| anyhow::anyhow!("failed to generate mobile invite id"))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(bytes))
}

fn invite_hash(invite_id: &str) -> String {
    let digest = ring::digest::digest(&ring::digest::SHA256, invite_id.as_bytes());
    digest
        .as_ref()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn validate_display_name(value: Option<&str>) -> anyhow::Result<String> {
    let value = value.unwrap_or("Workjet mobile pairing").trim();
    anyhow::ensure!(!value.is_empty(), "mobile invite display name is required");
    anyhow::ensure!(
        value.chars().count() <= 256,
        "mobile invite display name exceeds 256 characters"
    );
    anyhow::ensure!(
        !value.chars().any(char::is_control),
        "mobile invite display name contains a control character"
    );
    Ok(value.to_string())
}

pub fn create(root: &Path, ttl_seconds: i64, display_name: Option<&str>) -> anyhow::Result<Value> {
    anyhow::ensure!(
        (MIN_TTL_SECONDS..=MAX_TTL_SECONDS).contains(&ttl_seconds),
        "mobile invite ttlSeconds must be between {MIN_TTL_SECONDS} and {MAX_TTL_SECONDS}"
    );
    let display_name = validate_display_name(display_name)?;
    let created_at_ms = now_ms();
    let expires_at_ms = created_at_ms
        .checked_add(ttl_seconds * 1_000)
        .context("mobile invite expiry overflow")?;
    let expires_at = chrono::DateTime::from_timestamp_millis(expires_at_ms)
        .context("mobile invite expiry is invalid")?
        .to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
    let invite_id = random_invite_id()?;
    let invite_id_hash = invite_hash(&invite_id);
    let user_id = format!("workjet-mobile-invite-{}", &invite_id_hash[..24]);
    let (capability_token, capability_expires_at_ms) =
        super::store::issue_business_os_capability_token_for_managed_user_until(
            root,
            &user_id,
            "Workjet Mobile",
            "user",
            created_at_ms,
            expires_at_ms,
        )?;
    let config = super::store::sync_config(root)?;
    let conn = super::store::open_store(root)?;
    ensure_table(&conn)?;
    if let Err(error) = conn.execute(
        "INSERT INTO business_mobile_invites
            (invite_id_hash, user_id, created_at_ms, expires_at_ms, revoked_at_ms)
         VALUES (?1, ?2, ?3, ?4, NULL)",
        params![invite_id_hash, user_id, created_at_ms, expires_at_ms],
    ) {
        let _ = conn.execute(
            "UPDATE business_users
             SET active=0, capability_epoch=capability_epoch+1, updated_at_ms=?2
             WHERE user_id=?1",
            params![user_id, created_at_ms],
        );
        return Err(error).context("failed to persist mobile invite");
    }

    let invite = json!({
        "type": "ctox-business-os-invite",
        "version": 1,
        "display_name": display_name,
        "instance_id": config.instance_id,
        "sync_room": config.sync_room,
        "native_peer_id": config.native_peer_id,
        "signaling_urls": config.signaling_urls,
        "signaling_room_password": config.signaling_room_password,
        "transport": "webrtc",
        "expires_at": expires_at,
        "data_plane": "rxdb-webrtc",
        "http_bridge_available": false,
        "secret_value_in_payload": true,
        "session": {
            "authenticated": true,
            "source": "mobile_invite",
            "capability_token": capability_token,
            "capability_expires_at_ms": capability_expires_at_ms,
            "user": {
                "id": user_id,
                "display_name": "Workjet Mobile",
                "role": "user",
                "is_admin": false
            }
        }
    });
    Ok(json!({
        "inviteId": invite_id,
        "invite": invite,
        "expiresAt": expires_at
    }))
}

pub fn revoke(root: &Path, invite_id: &str) -> anyhow::Result<Value> {
    let invite_id = invite_id.trim();
    anyhow::ensure!(
        !invite_id.is_empty() && invite_id.len() <= 256,
        "mobile invite id is invalid"
    );
    let invite_id_hash = invite_hash(invite_id);
    let now = now_ms();
    let mut conn = super::store::open_store(root)?;
    ensure_table(&conn)?;
    let tx = conn.transaction()?;
    let user_id: Option<String> = tx
        .query_row(
            "SELECT user_id FROM business_mobile_invites WHERE invite_id_hash=?1",
            params![invite_id_hash],
            |row| row.get(0),
        )
        .optional()?;
    if let Some(user_id) = user_id {
        tx.execute(
            "UPDATE business_mobile_invites
             SET revoked_at_ms=COALESCE(revoked_at_ms, ?2)
             WHERE invite_id_hash=?1",
            params![invite_id_hash, now],
        )?;
        tx.execute(
            "UPDATE business_users
             SET active=0, capability_epoch=capability_epoch+1, updated_at_ms=?2
             WHERE user_id=?1 AND active=1",
            params![user_id, now],
        )?;
    }
    tx.commit()?;
    Ok(json!({ "revoked": true }))
}

#[cfg(test)]
mod tests {
    use super::{create, revoke};
    use tempfile::tempdir;

    #[test]
    fn mobile_invite_is_short_lived_and_individually_revocable() -> anyhow::Result<()> {
        let root = tempdir()?;
        let created = create(root.path(), 300, Some("Operations"))?;
        let invite = &created["invite"];
        let token = invite["session"]["capability_token"]
            .as_str()
            .expect("capability token");
        assert_eq!(invite["data_plane"], "rxdb-webrtc");
        assert_eq!(invite["http_bridge_available"], false);
        assert_eq!(invite["session"]["user"]["role"], "user");
        assert!(super::super::store::verify_capability_role(root.path(), token).is_some());

        let invite_id = created["inviteId"].as_str().expect("invite id");
        assert_eq!(
            revoke(root.path(), invite_id)?,
            serde_json::json!({ "revoked": true })
        );
        assert_eq!(
            revoke(root.path(), invite_id)?,
            serde_json::json!({ "revoked": true })
        );
        assert!(super::super::store::verify_capability_role(root.path(), token).is_none());
        assert_eq!(
            revoke(root.path(), "unknown-invite")?,
            serde_json::json!({ "revoked": true })
        );
        Ok(())
    }

    #[test]
    fn mobile_invite_ttl_is_bounded() {
        let root = tempdir().unwrap();
        assert!(create(root.path(), 59, None).is_err());
        assert!(create(root.path(), 3_601, None).is_err());
    }
}
